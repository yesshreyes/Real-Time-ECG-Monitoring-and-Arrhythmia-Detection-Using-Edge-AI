package com.example.rhythmai.presentation.stats

import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class StatsViewModel : ViewModel() {
    private val _rhythm = MutableStateFlow("Normal")
    val rhythm = _rhythm.asStateFlow()

    private val _confidence = MutableStateFlow(94)
    val confidence = _confidence.asStateFlow()
}
