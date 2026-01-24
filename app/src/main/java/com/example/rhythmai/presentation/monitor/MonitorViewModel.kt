package com.example.rhythmai.presentation.monitor

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.view.View
import androidx.core.view.drawToBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.rhythmai.data.EcgAssetLoader
import com.example.rhythmai.data.EcgTwoStageClassifier
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class MonitorViewModel(
    private val context: Context
) : ViewModel() {

    /* ---------------- UI STATE ---------------- */

    private val _ecgSamples = MutableStateFlow<List<Float>>(emptyList())
    val ecgSamples = _ecgSamples.asStateFlow()

    private val _bpm = MutableStateFlow(72)
    val bpm = _bpm.asStateFlow()

    private val _isAbnormal = MutableStateFlow(false)
    val isAbnormal = _isAbnormal.asStateFlow()

    private val _isLoading = MutableStateFlow(true)
    val isLoading = _isLoading.asStateFlow()

    private val _prediction = MutableStateFlow("Normal")
    val prediction = _prediction.asStateFlow()

    private val _confidence = MutableStateFlow(0f)
    val confidence = _confidence.asStateFlow()

    /* ---------------- INTERNAL STATE ---------------- */

    private val maxSamples = 360 * 6
    private var monitoringJob: Job? = null

    // ECG data
    private var ecgData: List<Float> = emptyList()
    private var ecgIndex = 0

    // BPM detection
    private var lastPeakTime = 0L
    private var lastSample = 0f

    //SLIDING WINDOW FOR ML
    private val inferenceWindowSize = 360
    private val inferenceBuffer = ArrayDeque<Float>()

    // TFLite
    private lateinit var classifier: EcgTwoStageClassifier

    private var lastAbnormalState = false

    private val _alertLatched = MutableStateFlow(false)
    val alertLatched = _alertLatched.asStateFlow()

    private val CONFIDENCE_THRESHOLD = 0.3f
    private val REQUIRED_CONSECUTIVE_ABNORMAL = 3

    private var consecutiveAbnormalCount = 0


    private val _stage1Threshold = MutableStateFlow(0.5f)
    val stage1Threshold = _stage1Threshold.asStateFlow()
    fun updateStage1Threshold(newThreshold: Float) {
        _stage1Threshold.value = newThreshold.coerceIn(0.1f, 0.9f)
        // Recreate classifier with new threshold
    }

    /* ---------------- LIFECYCLE ---------------- */

    fun startMonitoring() {
        if (monitoringJob != null) return

        monitoringJob = viewModelScope.launch {

            // Load ECG + ML model off main thread
            if (ecgData.isEmpty()) {
                withContext(Dispatchers.IO) {
                    ecgData = EcgAssetLoader.loadEcg(
                        context = context,
                        fileName = "200_ekg.csv",
                        leadName = "MLII"
                    )
                    classifier = EcgTwoStageClassifier(context)
                }
            }

            _isLoading.value = false

            while (true) {
                delay(3) // ~360 Hz

                val sample = ecgData[ecgIndex]
                ecgIndex = (ecgIndex + 1) % ecgData.size

                // UI waveform buffer
                _ecgSamples.value = (_ecgSamples.value + sample)
                    .takeLast(maxSamples)

                // BPM (for display)
                detectRPeak(sample)

                // ML sliding window
                inferenceBuffer.addLast(sample)
                if (inferenceBuffer.size > inferenceWindowSize) {
                    inferenceBuffer.removeFirst()
                }

                // Run inference every ~0.5s
                if (inferenceBuffer.size == inferenceWindowSize && ecgIndex % 60 == 0) {
                    runInference()
                }
            }
        }
    }

    fun stopMonitoring() {
        monitoringJob?.cancel()
        monitoringJob = null
    }

    /* ---------------- BPM ---------------- */

    private fun detectRPeak(sample: Float) {
        val threshold = 1.0f

        if (lastSample < threshold && sample >= threshold) {
            val now = System.currentTimeMillis()

            if (lastPeakTime != 0L) {
                val interval = now - lastPeakTime
                val bpmValue = (60000 / interval).toInt().coerceIn(40, 180)

                _bpm.value = bpmValue
            }

            lastPeakTime = now
        }

        lastSample = sample
    }

    /* ---------------- ML INFERENCE ---------------- */

    private fun runInference() {
        val window = inferenceBuffer.toFloatArray()

        // Use two-stage prediction
        val (result, conf) = classifier.predict(window)
        _prediction.value = result
        _confidence.value = conf
        // Better abnormal detection logic
        val abnormalWithConfidence =
            result != "Normal" && conf >= CONFIDENCE_THRESHOLD
        if (abnormalWithConfidence) {
            consecutiveAbnormalCount++
        } else {
            consecutiveAbnormalCount = 0
        }
        if (
            consecutiveAbnormalCount >= REQUIRED_CONSECUTIVE_ABNORMAL &&
            !_alertLatched.value
        ) {
            _alertLatched.value = true
            _isAbnormal.value = true
            triggerVibration()
            logArrhythmiaEvent(result, conf)  // Pass result type
        }
    }


    private fun triggerVibration() {
        val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE)
                as android.os.Vibrator

        if (android.os.Build.VERSION.SDK_INT >= 26) {
            vibrator.vibrate(
                android.os.VibrationEffect.createOneShot(
                    500,
                    android.os.VibrationEffect.DEFAULT_AMPLITUDE
                )
            )
        } else {
            vibrator.vibrate(500)
        }
    }

    private val _events =
        MutableStateFlow<List<ArrhythmiaEvent>>(emptyList())
    val events = _events.asStateFlow()

    private fun logArrhythmiaEvent(arrhythmiaType: String, confidence: Float) {
        val event = ArrhythmiaEvent(
            timestamp = System.currentTimeMillis(),
            arrhythmiaType = arrhythmiaType,  // New field
            confidence = confidence
        )
        _events.value = _events.value + event
    }

    fun clearAlert() {
        _alertLatched.value = false
    }

    fun openEmergencyDialer(context: Context) {
        val intent = Intent(Intent.ACTION_DIAL).apply {
            data = Uri.parse("tel:112")
        }
        context.startActivity(intent)
    }

    suspend fun captureEcgFromView(view: View) {
        withContext(Dispatchers.Main) {
            val bitmap = view.drawToBitmap()
            saveBitmapToGallery(bitmap)
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap) {
        val filename = "ECG_${System.currentTimeMillis()}.png"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/RhythmAI")
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            values
        ) ?: return

        resolver.openOutputStream(uri)?.use { out ->
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
    }



}
data class ArrhythmiaEvent(
    val timestamp: Long,
    val arrhythmiaType: String,
    val confidence: Float
)

