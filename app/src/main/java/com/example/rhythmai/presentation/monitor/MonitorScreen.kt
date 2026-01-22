package com.example.rhythmai.presentation.monitor

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.platform.LocalLifecycleOwner

@Composable
fun MonitorScreen(
    onStatsClick: () -> Unit,
    viewModel: MonitorViewModel = viewModel()
) {
    val bpm by viewModel.bpm.collectAsState()
    val isAbnormal by viewModel.isAbnormal.collectAsState()
    val samples by viewModel.ecgSamples.collectAsState()

    val lifecycleOwner = LocalLifecycleOwner.current

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> viewModel.startMonitoring()
                Lifecycle.Event.ON_STOP -> viewModel.stopMonitoring()
                else -> Unit
            }
        }

        lifecycleOwner.lifecycle.addObserver(observer)

        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {

        Text("Live ECG Monitor", fontSize = 22.sp)

        Spacer(Modifier.height(16.dp))

        EcgWaveform(samples = samples)

        Spacer(Modifier.height(16.dp))

        Text("Heart Rate: $bpm BPM", fontSize = 20.sp)

        Spacer(Modifier.height(8.dp))

        if (isAbnormal) {
            Text(
                "⚠ Arrhythmia Detected",
                color = Color.Red,
                fontSize = 18.sp
            )

            Spacer(Modifier.height(8.dp))

            Button(
                onClick = {},
                colors = ButtonDefaults.buttonColors(Color.Red)
            ) {
                Text("SOS", color = Color.White)
            }
        }

        Spacer(Modifier.weight(1f))

        Button(onClick = onStatsClick) {
            Text("View Stats")
        }
    }
}
