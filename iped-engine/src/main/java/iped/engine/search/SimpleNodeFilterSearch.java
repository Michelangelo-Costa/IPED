package iped.engine.search;

import java.math.BigDecimal;
import java.text.NumberFormat;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.lucene.search.BooleanClause.Occur;
import org.apache.lucene.search.BooleanQuery.Builder;
import org.apache.lucene.search.Query;

import iped.data.IIPEDSource;
import iped.engine.data.SimpleFilterNode;
import iped.exception.ParseException;
import iped.exception.QueryNodeException;
import iped.utils.LocalizedFormat;

public class SimpleNodeFilterSearch {
    private static final Map<SimpleFilterNode, Query> queryFromNode = Collections
            .synchronizedMap(new HashMap<SimpleFilterNode, Query>());

    private static final Pattern RANGE_PATTERN = Pattern.compile("([\\[{])\\s*(\\S+)\\s+TO\\s+(\\S+)\\s*([\\]}])"); //$NON-NLS-1$

    public static Query getNodeQuery(IIPEDSource ipedSource, SimpleFilterNode node) {
        Query query = queryFromNode.get(node);
        if (query == null) {
            String property = node.getProperty();
            if (property != null) {
                String value = node.getValue();
                if (value != null) {
                    value = value.trim();
                    property = property.trim();
                    StringBuffer queryStr = new StringBuffer();
                    queryStr.append(property);
                    queryStr.append(":");
                    if (value.length() > 1 && !isRangeExpression(value)) {
                        queryStr.append('"');
                        queryStr.append(value);
                        queryStr.append('"');
                    } else {
                        queryStr.append(localizeRangeBounds(value));
                    }
                    try {
                        query = new QueryBuilder(ipedSource).getQuery(queryStr.toString());
                    } catch (ParseException | QueryNodeException e) {
                        e.printStackTrace();
                    }
                }
            }
            if (node.getAddChildren()) {
                Builder builder = new Builder();
                if (query != null) {
                    builder.add(query, Occur.SHOULD);
                }
                for (SimpleFilterNode child : node.getChildren()) {
                    Query childQuery = getNodeQuery(ipedSource, child);
                    if (childQuery != null) {
                        builder.add(childQuery, Occur.SHOULD);
                    }
                }
                query = builder.build();
            }
            if (query != null) {
                queryFromNode.put(node, query);
            }
        }
        return query;
    }

    /**
     * Checks if a filter value is a range expression, which must not be quoted.
     * Accepts both inclusive ("]") and exclusive ("}") upper bounds, so contiguous
     * ranges over float properties can be declared, e.g. "[0.40 TO 0.70}".
     */
    private static boolean isRangeExpression(String value) {
        return value.endsWith("]") || value.endsWith("}");
    }

    /**
     * Range bounds are declared in the configuration with a locale independent
     * format (dot as decimal separator), but {@link QueryBuilder} parses numbers
     * with the localized format of the UI. Converts numeric bounds to that format,
     * otherwise "0.70" would be parsed as 70 in locales that use dot as grouping
     * separator (e.g. pt-BR). Non numeric bounds, like "*", are kept unchanged.
     */
    private static String localizeRangeBounds(String value) {
        Matcher matcher = RANGE_PATTERN.matcher(value);
        if (!matcher.matches()) {
            return value;
        }
        return matcher.group(1) + localizeNumber(matcher.group(2)) + " TO " + localizeNumber(matcher.group(3)) //$NON-NLS-1$
                + matcher.group(4);
    }

    private static String localizeNumber(String bound) {
        BigDecimal number;
        try {
            number = new BigDecimal(bound);
        } catch (NumberFormatException e) {
            return bound;
        }
        NumberFormat nf = LocalizedFormat.getNumberInstance();
        nf.setGroupingUsed(false);
        nf.setMaximumFractionDigits(Math.max(0, number.scale()));
        return nf.format(number);
    }
}
