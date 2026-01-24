package com.example.rhythmai.presentation.stats

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import androidx.lifecycle.viewModelScope
import com.example.rhythmai.presentation.monitor.ArrhythmiaEvent
import com.example.rhythmai.presentation.monitor.MonitorViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch

class StatsViewModel(
    monitorViewModel: MonitorViewModel
) : ViewModel() {

    /* ---------------- ARRHYTHMIA EVENTS ---------------- */

    val events: StateFlow<List<ArrhythmiaEvent>> =
        monitorViewModel.events
            .stateIn(
                scope = viewModelScope,
                started = SharingStarted.WhileSubscribed(5_000),
                initialValue = emptyList()
            )

    /* ---------------- STATIC MODEL INFO ---------------- */

    val modelArchitecture = listOf(
        "Type" to "Two-Stage Cascade",
        "Stage 1" to "Normal vs Abnormal (Screening)",
        "Stage 2" to "SV vs Ventricular (Diagnosis)",
        "Total Model Size" to "≈ 1 MB"
    )

    val modelPerformance = listOf(
        "Dataset" to "MIT-BIH Arrhythmia",
        "Stage 1 Sensitivity" to "≥ 95%",
        "Stage 2 F1-Score" to "≥ 90%",
        "Validation Method" to "Inter-patient split"
    )
}
