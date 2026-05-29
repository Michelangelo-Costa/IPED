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
package iped.aigcd;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.imageio.ImageIO;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import iped.configuration.Configurable;
import iped.data.IItem;
import iped.engine.config.Configuration;
import iped.engine.config.ConfigurationManager;
import iped.engine.config.EnableTaskProperty;
import iped.engine.task.AbstractTask;

/**
 * Detects AI-generated images using a ConvNeXt V2 Base ONNX model.
 *
 * Model: https://huggingface.co/xRayon/convnext-ai-images-detector
 *
 * Place the model at: &lt;iped-root&gt;/models/convnext-ai-detector/model.onnx.
 * Score is stored as aigcd:score:image (0.0 = human/real, 1.0 = AI-generated).
 */
public class AIGCDTask extends AbstractTask {

    public static final String ENABLE_PARAM = "enableAIGCDDetector"; //$NON-NLS-1$
    public static final String SCORE_IMAGE_FIELD = "aigcd:score:image"; //$NON-NLS-1$

    private static final String MODEL_RELATIVE_PATH = "models/convnext-ai-detector/model.onnx"; //$NON-NLS-1$
    private static final int RESIZE_SIZE = 288;
    private static final int CROP_SIZE = 256;
    private static final float[] MEAN = { 0.485f, 0.456f, 0.406f };
    private static final float[] STD  = { 0.229f, 0.224f, 0.225f };

    private static final Logger logger = LoggerFactory.getLogger(AIGCDTask.class);

    private static final AtomicBoolean init     = new AtomicBoolean(false);
    private static final AtomicBoolean finished = new AtomicBoolean(false);

    private static volatile boolean taskEnabled = false;
    private static volatile boolean modelLoaded = false;
    private static volatile OrtEnvironment env;
    private static volatile OrtSession session;

    @Override
    public boolean isEnabled() {
        return taskEnabled;
    }

    @Override
    public List<Configurable<?>> getConfigurables() {
        return Arrays.asList(new EnableTaskProperty(ENABLE_PARAM));
    }

    @Override
    public void init(ConfigurationManager configurationManager) throws Exception {
        synchronized (init) {
            if (init.get()) {
                return;
            }
            taskEnabled = configurationManager.getEnableTaskProperty(ENABLE_PARAM);
            if (!taskEnabled) {
                logger.info("Task disabled."); //$NON-NLS-1$
                init.set(true);
                return;
            }

            File modelFile = new File(Configuration.getInstance().appRoot, MODEL_RELATIVE_PATH);
            if (!modelFile.exists()) {
                logger.warn("convnext-ai-detector model not found at {}. Task will be skipped.", //$NON-NLS-1$
                        modelFile.getAbsolutePath());
                init.set(true);
                return;
            }

            try {
                env = OrtEnvironment.getEnvironment();
                session = env.createSession(modelFile.getAbsolutePath(), new OrtSession.SessionOptions());
                modelLoaded = true;
                logger.info("Model loaded from {}", modelFile.getAbsolutePath()); //$NON-NLS-1$
            } catch (Exception e) {
                logger.error("Failed to load model.", e); //$NON-NLS-1$
            }
            init.set(true);
        }
    }

    @Override
    public void finish() throws Exception {
        synchronized (finished) {
            if (!finished.get()) {
                if (session != null) {
                    session.close();
                    session = null;
                }
                if (env != null) {
                    env.close();
                    env = null;
                }
                finished.set(true);
            }
        }
    }

    @Override
    protected void process(IItem item) throws Exception {
        if (!taskEnabled || !modelLoaded || !item.isToAddToCase() || !isImage(item)) {
            return;
        }

        try (InputStream stream = item.getBufferedInputStream()) {
            if (stream == null) {
                return;
            }
            BufferedImage image = ImageIO.read(stream);
            if (image == null) {
                return;
            }
            float score = runInference(image);
            item.setExtraAttribute(SCORE_IMAGE_FIELD, score);
        } catch (Exception e) {
            logger.warn("Error processing item {}: {}", item.getName(), e.getMessage()); //$NON-NLS-1$
        }
    }

    private boolean isImage(IItem item) {
        String mime = item.getMediaType() != null ? item.getMediaType().toString() : ""; //$NON-NLS-1$
        return mime.startsWith("image/"); //$NON-NLS-1$
    }

    private float runInference(BufferedImage original) throws Exception {
        BufferedImage preprocessed = resizeAndCrop(original, RESIZE_SIZE, CROP_SIZE);
        float[] tensorData = toNormalizedTensor(preprocessed);

        long[] shape = { 1, 3, CROP_SIZE, CROP_SIZE };
        try (OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape);
                OrtSession.Result result = session.run(
                        Collections.singletonMap(session.getInputNames().iterator().next(), tensor))) {

            float[][] logits = (float[][]) result.get(0).getValue();
            // Model output: [0] = real score, [1] = AI-generated score
            float expReal = (float) Math.exp(logits[0][0]);
            float expAI   = (float) Math.exp(logits[0][1]);
            return expAI / (expReal + expAI);
        }
    }

    private BufferedImage resizeAndCrop(BufferedImage src, int resizeTo, int cropSize) {
        int w = src.getWidth();
        int h = src.getHeight();
        int newW, newH;
        if (w < h) {
            newW = resizeTo;
            newH = (int) Math.round((double) h * resizeTo / w);
        } else {
            newH = resizeTo;
            newW = (int) Math.round((double) w * resizeTo / h);
        }
        BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.drawImage(src, 0, 0, newW, newH, null);
        g.dispose();

        int x0 = (newW - cropSize) / 2;
        int y0 = (newH - cropSize) / 2;
        return resized.getSubimage(x0, y0, cropSize, cropSize);
    }

    private float[] toNormalizedTensor(BufferedImage img) {
        float[] tensor = new float[3 * CROP_SIZE * CROP_SIZE];
        for (int y = 0; y < CROP_SIZE; y++) {
            for (int x = 0; x < CROP_SIZE; x++) {
                int rgb = img.getRGB(x, y);
                float r = ((rgb >> 16) & 0xFF) / 255.0f;
                float g = ((rgb >> 8)  & 0xFF) / 255.0f;
                float b = ( rgb        & 0xFF) / 255.0f;

                int idx = y * CROP_SIZE + x;
                tensor[idx]                          = (r - MEAN[0]) / STD[0];
                tensor[CROP_SIZE * CROP_SIZE + idx]  = (g - MEAN[1]) / STD[1];
                tensor[2 * CROP_SIZE * CROP_SIZE + idx] = (b - MEAN[2]) / STD[2];
            }
        }
        return tensor;
    }
}
