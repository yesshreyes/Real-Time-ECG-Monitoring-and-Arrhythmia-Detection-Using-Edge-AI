package com.example.rhythmai.data
import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
class EcgTwoStageClassifier(context: Context) {
    private val stage1Interpreter: Interpreter
    private val stage2Interpreter: Interpreter
    // Configurable threshold for Stage 1
    private val stage1Threshold = 0.5f
    init {
        // Load Stage 1 Model (Normal vs Abnormal)
        val stage1ModelName = "stage1_model.tflite"
        Log.d("TFLite", "Loading Stage 1 model: $stage1ModelName")

        val stage1Bytes = context.assets.open(stage1ModelName).readBytes()
        val stage1Buffer = ByteBuffer.allocateDirect(stage1Bytes.size)
            .order(ByteOrder.nativeOrder())
        stage1Buffer.put(stage1Bytes)

        stage1Interpreter = Interpreter(stage1Buffer)

        // Load Stage 2 Model (SV vs Ventricular)
        val stage2ModelName = "stage2_model.tflite"
        Log.d("TFLite", "Loading Stage 2 model: $stage2ModelName")

        val stage2Bytes = context.assets.open(stage2ModelName).readBytes()
        val stage2Buffer = ByteBuffer.allocateDirect(stage2Bytes.size)
            .order(ByteOrder.nativeOrder())
        stage2Buffer.put(stage2Bytes)

        stage2Interpreter = Interpreter(stage2Buffer)

        Log.d("TFLite", "Both models loaded successfully")
    }
    /**
     * Two-stage prediction pipeline
     * @param window ECG window (360 samples)
     * @return Pair of (label, confidence)
     */
    fun predict(window: FloatArray): Pair<String, Float> {

        // Prepare input: [1, 360, 1]
        val input = Array(1) {
            Array(window.size) { floatArrayOf(0f) }
        }

        for (i in window.indices) {
            input[0][i][0] = window[i]
        }

        // ========== STAGE 1: Normal vs Abnormal ==========

        // Output shape: [1, 1] (sigmoid probability)
        val stage1Output = Array(1) { FloatArray(1) }

        stage1Interpreter.run(input, stage1Output)

        val abnormalProb = stage1Output[0][0]

        Log.d("TFLite", "Stage 1 - Abnormal probability: $abnormalProb")

        // If Normal (below threshold), return immediately
        if (abnormalProb < stage1Threshold) {
            val normalConfidence = 1f - abnormalProb
            Log.d("TFLite", "Classification: Normal (confidence: $normalConfidence)")
            return "Normal" to normalConfidence
        }

        // ========== STAGE 2: SV vs Ventricular ==========

        Log.d("TFLite", "Abnormal detected, running Stage 2...")

        // Output shape: [1, 2] (softmax probabilities)
        val stage2Output = Array(1) { FloatArray(2) }

        stage2Interpreter.run(input, stage2Output)

        val svProb = stage2Output[0][0]
        val ventProb = stage2Output[0][1]

        Log.d("TFLite", "Stage 2 - SV: $svProb, Ventricular: $ventProb")

        // Determine final class
        return if (svProb > ventProb) {
            "Supraventricular Arrhythmia" to svProb
        } else {
            "Ventricular Arrhythmia" to ventProb
        }
    }

    /**
     * Get detailed prediction with all probabilities
     */
    fun predictDetailed(window: FloatArray): PredictionResult {

        val input = Array(1) {
            Array(window.size) { floatArrayOf(0f) }
        }

        for (i in window.indices) {
            input[0][i][0] = window[i]
        }

        // Stage 1
        val stage1Output = Array(1) { FloatArray(1) }
        stage1Interpreter.run(input, stage1Output)
        val abnormalProb = stage1Output[0][0]

        if (abnormalProb < stage1Threshold) {
            return PredictionResult(
                finalClass = "Normal",
                confidence = 1f - abnormalProb,
                stage1AbnormalProb = abnormalProb,
                stage2SvProb = null,
                stage2VentProb = null
            )
        }

        // Stage 2
        val stage2Output = Array(1) { FloatArray(2) }
        stage2Interpreter.run(input, stage2Output)

        val svProb = stage2Output[0][0]
        val ventProb = stage2Output[0][1]

        val (finalClass, confidence) = if (svProb > ventProb) {
            "Supraventricular Arrhythmia" to svProb
        } else {
            "Ventricular Arrhythmia" to ventProb
        }

        return PredictionResult(
            finalClass = finalClass,
            confidence = confidence,
            stage1AbnormalProb = abnormalProb,
            stage2SvProb = svProb,
            stage2VentProb = ventProb
        )
    }

    fun close() {
        stage1Interpreter.close()
        stage2Interpreter.close()
    }
}
/**
 * Detailed prediction result with all intermediate probabilities
 */
data class PredictionResult(
    val finalClass: String,
    val confidence: Float,
    val stage1AbnormalProb: Float,
    val stage2SvProb: Float?,
    val stage2VentProb: Float?
)