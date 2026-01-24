package com.example.rhythmai.data
import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
/**
 * Three-Stage ECG Arrhythmia Classifier with Signal Quality Index (SQI)
 *
 * Pipeline:
 * 1. SQI Model: Check signal quality (GOOD vs BAD)
 * 2. Stage 1 Model: Screen for abnormalities (Normal vs Abnormal)
 * 3. Stage 2 Model: Diagnose arrhythmia type (SV vs Ventricular)
 */
class EcgThreeStageClassifier(context: Context) {

    private val sqiInterpreter: Interpreter
    private val stage1Interpreter: Interpreter
    private val stage2Interpreter: Interpreter

    // Configurable thresholds
    private val sqiThreshold = 0.01f        // Quality score threshold
    private val stage1Threshold = 0.5f     // Abnormality threshold

    init {
        // Load SQI Model (Signal Quality Index)
        val sqiModelName = "sqi_model.tflite"
        Log.d("TFLite", "Loading SQI model: $sqiModelName")

        val sqiBytes = context.assets.open(sqiModelName).readBytes()
        val sqiBuffer = ByteBuffer.allocateDirect(sqiBytes.size)
            .order(ByteOrder.nativeOrder())
        sqiBuffer.put(sqiBytes)

        sqiInterpreter = Interpreter(sqiBuffer)

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

        Log.d("TFLite", "All three models loaded successfully (SQI + Stage 1 + Stage 2)")
    }

    /**
     * Three-stage prediction pipeline with quality check
     * @param window ECG window (360 samples)
     * @return Pair of (label, confidence)
     */
    fun predict(window: FloatArray): Pair<String, Float> {
        val result = predictDetailed(window)

        // If poor quality, return quality message
        if (result.qualityStatus == QualityStatus.BAD) {
            return result.message!! to 0f
        }

        return result.finalClass to result.confidence
    }

    /**
     * Get detailed prediction with all probabilities and quality assessment
     */
    fun predictDetailed(window: FloatArray): PredictionResult {

        // Prepare input: [1, 360, 1]
        val input = Array(1) {
            Array(window.size) { floatArrayOf(0f) }
        }

        for (i in window.indices) {
            input[0][i][0] = window[i]
        }

        // ========== SQI: Signal Quality Check ==========

        val sqiOutput = Array(1) { FloatArray(1) }
        sqiInterpreter.run(input, sqiOutput)

        val qualityScore = sqiOutput[0][0]

        Log.d("TFLite", "SQI - Quality score: $qualityScore")

        // If signal quality is poor, skip inference
        if (qualityScore < sqiThreshold) {
            Log.d("TFLite", "Poor signal quality detected. Skipping inference.")
            return PredictionResult(
                qualityScore = qualityScore,
                qualityStatus = QualityStatus.BAD,
                finalClass = "Unknown",
                confidence = 0f,
                message = "Poor signal quality. Please adjust sensor.",
                stage1AbnormalProb = null,
                stage2SvProb = null,
                stage2VentProb = null
            )
        }

        Log.d("TFLite", "Signal quality is GOOD. Proceeding to Stage 1...")

        // ========== STAGE 1: Normal vs Abnormal ==========

        val stage1Output = Array(1) { FloatArray(1) }
        stage1Interpreter.run(input, stage1Output)

        val abnormalProb = stage1Output[0][0]

        Log.d("TFLite", "Stage 1 - Abnormal probability: $abnormalProb")

        // If Normal (below threshold), return immediately
        if (abnormalProb < stage1Threshold) {
            val normalConfidence = 1f - abnormalProb
            Log.d("TFLite", "Classification: Normal (confidence: $normalConfidence)")
            return PredictionResult(
                qualityScore = qualityScore,
                qualityStatus = QualityStatus.GOOD,
                finalClass = "Normal",
                confidence = normalConfidence,
                message = null,
                stage1AbnormalProb = abnormalProb,
                stage2SvProb = null,
                stage2VentProb = null
            )
        }

        // ========== STAGE 2: SV vs Ventricular ==========

        Log.d("TFLite", "Abnormal detected, running Stage 2...")

        val stage2Output = Array(1) { FloatArray(2) }
        stage2Interpreter.run(input, stage2Output)

        val svProb = stage2Output[0][0]
        val ventProb = stage2Output[0][1]

        Log.d("TFLite", "Stage 2 - SV: $svProb, Ventricular: $ventProb")

        // Determine final class
        val (finalClass, confidence) = if (svProb > ventProb) {
            "Supraventricular Arrhythmia" to svProb
        } else {
            "Ventricular Arrhythmia" to ventProb
        }

        return PredictionResult(
            qualityScore = qualityScore,
            qualityStatus = QualityStatus.GOOD,
            finalClass = finalClass,
            confidence = confidence,
            message = null,
            stage1AbnormalProb = abnormalProb,
            stage2SvProb = svProb,
            stage2VentProb = ventProb
        )
    }

    /**
     * Check only signal quality (useful for real-time quality monitoring)
     */
    fun checkQuality(window: FloatArray): Pair<Float, QualityStatus> {
        val input = Array(1) {
            Array(window.size) { floatArrayOf(0f) }
        }

        for (i in window.indices) {
            input[0][i][0] = window[i]
        }

        val sqiOutput = Array(1) { FloatArray(1) }
        sqiInterpreter.run(input, sqiOutput)

        val qualityScore = sqiOutput[0][0]
        val status = if (qualityScore >= sqiThreshold) {
            QualityStatus.GOOD
        } else {
            QualityStatus.BAD
        }

        return qualityScore to status
    }

    fun close() {
        sqiInterpreter.close()
        stage1Interpreter.close()
        stage2Interpreter.close()
    }
}
/**
 * Signal quality status
 */
enum class QualityStatus {
    GOOD,
    BAD
}
/**
 * Detailed prediction result with quality assessment and all intermediate probabilities
 */
data class PredictionResult(
    // Quality assessment
    val qualityScore: Float,
    val qualityStatus: QualityStatus,

    // Final prediction
    val finalClass: String,
    val confidence: Float,
    val message: String?,

    // Intermediate probabilities
    val stage1AbnormalProb: Float?,
    val stage2SvProb: Float?,
    val stage2VentProb: Float?
)