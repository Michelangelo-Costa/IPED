/*
 * This file is part of Indexador e Processador de Evidências Digitais (IPED).
 *
 * IPED is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * IPED is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with IPED.  If not, see <http://www.gnu.org/licenses/>.
 */
package iped.app.ui;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;

import javax.swing.BorderFactory;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexableField;

public class AIGCDPanel extends JPanel {

    private static final long serialVersionUID = 1L;

    private static final String SCORE_IMAGE_FIELD = "aigcd:score:image"; //$NON-NLS-1$

    /**
     * Lower bounds of the High and Medium levels. They must match the score ranges
     * of the "AIGCD - Image Score" filters in AIFiltersConfig.json, so the level
     * shown in this panel is always the one of the filter that lists the item.
     */
    private static final float HIGH_THRESHOLD = 0.70f;
    private static final float MEDIUM_THRESHOLD = 0.40f;

    private static final float NO_SCORE = -1f;

    private enum Level {
        NA("AIGCDPanel.NA", Color.LIGHT_GRAY, Color.GRAY), //$NON-NLS-1$
        LOW("AIGCDPanel.Low", new Color(60, 160, 60), new Color(40, 120, 40)), //$NON-NLS-1$
        MEDIUM("AIGCDPanel.Medium", new Color(220, 140, 0), new Color(180, 100, 0)), //$NON-NLS-1$
        HIGH("AIGCDPanel.High", new Color(200, 50, 50), new Color(200, 50, 50)); //$NON-NLS-1$

        private final String messageKey;
        private final Color color;
        private final Color textColor;

        Level(String messageKey, Color color, Color textColor) {
            this.messageKey = messageKey;
            this.color = color;
            this.textColor = textColor;
        }

        static Level of(float score) {
            if (Float.isNaN(score) || score < 0) {
                return NA;
            }
            if (score >= HIGH_THRESHOLD) {
                return HIGH;
            }
            if (score >= MEDIUM_THRESHOLD) {
                return MEDIUM;
            }
            return LOW;
        }
    }

    private JLabel overallLabel;
    private JPanel overallBadge;
    private JProgressBar imageBar;
    private JLabel imageScore;

    public AIGCDPanel() {
        setLayout(new BorderLayout(8, 8));
        setBorder(BorderFactory.createEmptyBorder(12, 12, 12, 12));

        add(buildHeader(), BorderLayout.NORTH);
        add(buildScores(), BorderLayout.CENTER);
        add(buildFooter(), BorderLayout.SOUTH);

        showEmpty();
    }

    private JPanel buildHeader() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));

        JLabel title = new JLabel(Messages.getString("AIGCDPanel.Title")); //$NON-NLS-1$
        title.setFont(title.getFont().deriveFont(Font.BOLD, 13f));
        panel.add(title);

        overallBadge = new JPanel();
        overallBadge.setPreferredSize(new Dimension(90, 22));
        overallBadge.setLayout(new BorderLayout());
        overallLabel = new JLabel("", SwingConstants.CENTER); //$NON-NLS-1$
        overallLabel.setFont(overallLabel.getFont().deriveFont(Font.BOLD, 11f));
        overallLabel.setForeground(Color.WHITE);
        overallBadge.add(overallLabel, BorderLayout.CENTER);
        panel.add(overallBadge);

        return panel;
    }

    private JPanel buildScores() {
        JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(Messages.getString("AIGCDPanel.ScoresByModality"))); //$NON-NLS-1$

        GridBagConstraints lc = new GridBagConstraints();
        lc.anchor = GridBagConstraints.WEST;
        lc.insets = new Insets(4, 6, 4, 8);
        lc.gridx = 0;

        GridBagConstraints bc = new GridBagConstraints();
        bc.fill = GridBagConstraints.HORIZONTAL;
        bc.insets = new Insets(4, 0, 4, 8);
        bc.gridx = 1;
        bc.weightx = 1.0;

        GridBagConstraints sc = new GridBagConstraints();
        sc.anchor = GridBagConstraints.EAST;
        sc.insets = new Insets(4, 0, 4, 6);
        sc.gridx = 2;

        imageBar   = makeBar();
        imageScore = makeScoreLabel();

        lc.gridy = bc.gridy = sc.gridy = 0;
        panel.add(new JLabel(Messages.getString("AIGCDPanel.Image") + ":"), lc); //$NON-NLS-1$
        panel.add(imageBar, bc);
        panel.add(imageScore, sc);

        return panel;
    }

    private JPanel buildFooter() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT));
        JLabel note = new JLabel("<html><i>" + Messages.getString("AIGCDPanel.Footer") + "</i></html>"); //$NON-NLS-1$ //$NON-NLS-2$ //$NON-NLS-3$
        note.setForeground(Color.GRAY);
        note.setFont(note.getFont().deriveFont(10f));
        panel.add(note);
        return panel;
    }

    private JProgressBar makeBar() {
        JProgressBar bar = new JProgressBar(0, 100);
        bar.setStringPainted(false);
        bar.setPreferredSize(new Dimension(200, 18));
        return bar;
    }

    private JLabel makeScoreLabel() {
        JLabel l = new JLabel(Messages.getString("AIGCDPanel.NA")); //$NON-NLS-1$
        l.setFont(l.getFont().deriveFont(Font.BOLD, 11f));
        l.setPreferredSize(new Dimension(55, 18));
        return l;
    }

    public void loadDoc(Document doc) {
        if (doc == null) {
            SwingUtilities.invokeLater(this::showEmpty);
            return;
        }
        float imgScore = parseScore(doc, SCORE_IMAGE_FIELD);
        SwingUtilities.invokeLater(() -> applyScore(imgScore));
    }

    private float parseScore(Document doc, String field) {
        String val = doc.get(field);
        if (val != null && !val.isEmpty()) {
            try {
                return Float.parseFloat(val);
            } catch (NumberFormatException e) {
                return NO_SCORE;
            }
        }
        IndexableField indexableField = doc.getField(field);
        if (indexableField == null || indexableField.numericValue() == null) {
            return NO_SCORE;
        }
        return indexableField.numericValue().floatValue();
    }

    /**
     * Truncates (instead of rounding) the score to a percentage, so the displayed
     * value never crosses a level boundary: a 0.6951 score is shown as 69% (Medium).
     */
    private int toPercent(float score) {
        return Math.max(0, Math.min(100, (int) (score * 100)));
    }

    private void applyScore(float score) {
        Level level = Level.of(score);
        setBar(imageBar, imageScore, score, level);
        overallLabel.setText(Messages.getString(level.messageKey));
        overallBadge.setBackground(level.color);
    }

    private void setBar(JProgressBar bar, JLabel label, float score, Level level) {
        bar.setForeground(level.color);
        label.setForeground(level.textColor);
        if (level == Level.NA) {
            bar.setValue(0);
            label.setText(Messages.getString(level.messageKey));
        } else {
            int percent = toPercent(score);
            bar.setValue(percent);
            label.setText(percent + "%"); //$NON-NLS-1$
        }
    }

    private void showEmpty() {
        applyScore(NO_SCORE);
    }
}
