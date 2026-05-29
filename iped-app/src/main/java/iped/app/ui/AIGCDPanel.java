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
        int imgScore = parseScore(doc, SCORE_IMAGE_FIELD);
        SwingUtilities.invokeLater(() -> applyScore(imgScore));
    }

    private int parseScore(Document doc, String field) {
        String val = doc.get(field);
        if (val != null && !val.isEmpty()) {
            try {
                return toPercent(Double.parseDouble(val));
            } catch (NumberFormatException e) {
                return -1;
            }
        }
        IndexableField indexableField = doc.getField(field);
        if (indexableField == null || indexableField.numericValue() == null) {
            return -1;
        }
        return toPercent(indexableField.numericValue().doubleValue());
    }

    private int toPercent(double score) {
        return Math.max(0, Math.min(100, (int) Math.round(score * 100)));
    }

    private void applyScore(int img) {
        setBar(imageBar, imageScore, img);

        if (img < 0) {
            overallLabel.setText(Messages.getString("AIGCDPanel.NA")); //$NON-NLS-1$
            overallBadge.setBackground(Color.LIGHT_GRAY);
        } else if (img >= 70) {
            overallLabel.setText(Messages.getString("AIGCDPanel.High")); //$NON-NLS-1$
            overallBadge.setBackground(new Color(200, 50, 50));
        } else if (img >= 40) {
            overallLabel.setText(Messages.getString("AIGCDPanel.Medium")); //$NON-NLS-1$
            overallBadge.setBackground(new Color(220, 140, 0));
        } else {
            overallLabel.setText(Messages.getString("AIGCDPanel.Low")); //$NON-NLS-1$
            overallBadge.setBackground(new Color(60, 160, 60));
        }
    }

    private void setBar(JProgressBar bar, JLabel label, int score) {
        if (score < 0) {
            bar.setValue(0);
            bar.setForeground(Color.LIGHT_GRAY);
            label.setText(Messages.getString("AIGCDPanel.NA")); //$NON-NLS-1$
            label.setForeground(Color.GRAY);
        } else {
            bar.setValue(score);
            label.setText(score + "%"); //$NON-NLS-1$
            if (score >= 70) {
                bar.setForeground(new Color(200, 50, 50));
                label.setForeground(new Color(200, 50, 50));
            } else if (score >= 40) {
                bar.setForeground(new Color(220, 140, 0));
                label.setForeground(new Color(180, 100, 0));
            } else {
                bar.setForeground(new Color(60, 160, 60));
                label.setForeground(new Color(40, 120, 40));
            }
        }
    }

    private void showEmpty() {
        setBar(imageBar, imageScore, -1);
        overallLabel.setText(Messages.getString("AIGCDPanel.NA")); //$NON-NLS-1$
        overallBadge.setBackground(Color.LIGHT_GRAY);
    }
}
