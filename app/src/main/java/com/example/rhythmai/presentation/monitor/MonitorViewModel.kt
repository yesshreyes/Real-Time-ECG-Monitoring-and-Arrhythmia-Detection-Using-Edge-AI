package com.example.rhythmai.presentation.monitor

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class MonitorViewModel : ViewModel() {

    private val _ecgSamples = MutableStateFlow<List<Float>>(emptyList())
    val ecgSamples = _ecgSamples.asStateFlow()

    private val _bpm = MutableStateFlow(72)
    val bpm = _bpm.asStateFlow()

    private val _isAbnormal = MutableStateFlow(false)
    val isAbnormal = _isAbnormal.asStateFlow()

    private val maxSamples = 300
    private var monitoringJob: Job? = null

    private var abnormalCountdown = 0
    private val abnormalDuration = 8
    private val abnormalChance = 0.02f

    fun startMonitoring() {
        if (monitoringJob != null) return

        monitoringJob = viewModelScope.launch {
            while (true) {
                delay(16)

                val nextSample = generateFakeEcgSample()
                _ecgSamples.value = (_ecgSamples.value + nextSample)
                    .takeLast(maxSamples)

                if (System.currentTimeMillis() % 1000L < 20L) {
                    updateHeartState()
                }
            }
        }
    }

    fun stopMonitoring() {
        monitoringJob?.cancel()
        monitoringJob = null
    }

    private fun updateHeartState() {
        if (abnormalCountdown > 0) {
            abnormalCountdown--
            _bpm.value = (105..130).random()
            _isAbnormal.value = true
        } else {
            // Randomly enter abnormal state
            if (Math.random() < abnormalChance) {
                abnormalCountdown = abnormalDuration
                _isAbnormal.value = true
            } else {
                _bpm.value = (65..85).random()
                _isAbnormal.value = false
            }
        }
    }

    private fun generateFakeEcgSample(): Float {
        val t = System.currentTimeMillis() / 80.0
        val base = kotlin.math.sin(t).toFloat()

        val spikeChance = if (_isAbnormal.value) 0.15 else 0.04
        val spike = if (Math.random() < spikeChance) 1.2f else 0f

        return base * 0.3f + spike
    }
}
