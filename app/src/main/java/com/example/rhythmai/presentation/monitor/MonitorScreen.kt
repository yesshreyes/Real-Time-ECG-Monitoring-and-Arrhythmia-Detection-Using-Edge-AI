package com.example.rhythmai.presentation.monitor

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun MonitorScreen(
    viewModel: MonitorViewModel,
    onStatsClick: () -> Unit
) {
    val isLoading by viewModel.isLoading.collectAsState()
    val bpm by viewModel.bpm.collectAsState()
    val samples by viewModel.ecgSamples.collectAsState()
    val prediction by viewModel.prediction.collectAsState()
    val confidence by viewModel.confidence.collectAsState()
    val isAbnormal by viewModel.isAbnormal.collectAsState()
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    val rootView = LocalView.current
    val rhythmColor = when (prediction) {
        "Normal" -> Color(0xFF2E7D32)
        "Supraventricular Arrhythmia" -> Color(0xFFF9A825)
        "Ventricular Arrhythmia" -> Color(0xFFC62828)
        else -> Color.White
    }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_START -> viewModel.startMonitoring()
                Lifecycle.Event.ON_STOP -> viewModel.stopMonitoring()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    if (isLoading) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                CircularProgressIndicator()
                Spacer(Modifier.height(12.dp))
                Text("Initializing ECG stream…")
            }
        }
        return
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {

        /* ---------- HEADER ---------- */

        Text(
            text = "Real-Time ECG Monitor",
            fontSize = 22.sp,
            style = MaterialTheme.typography.titleLarge
        )

        Spacer(Modifier.height(12.dp))

        /* ---------- ECG + STATS (CAPTURED AREA) ---------- */

        EcgWaveform(samples = samples)

        Spacer(Modifier.height(12.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text("Heart Rate", fontSize = 14.sp, color = Color.Gray)
                Text("$bpm BPM", fontSize = 28.sp)
            }

            IconButton(
                onClick = {
                    coroutineScope.launch {
                        viewModel.captureEcgFromView(rootView)
                    }
                }
            ) {
                Icon(
                    imageVector = Icons.Default.CameraAlt,
                    contentDescription = "Capture ECG",
                    tint = Color(0xFF2E7D32),
                    modifier = Modifier.size(28.dp)
                )
            }
        }

        Spacer(Modifier.height(8.dp))

        Text(
            text = prediction,
            fontSize = 18.sp,
            color = rhythmColor
        )

        Text(
            text = "Confidence: ${(confidence * 100).toInt()}%",
            fontSize = 12.sp,
            color = Color.Gray
        )

        Text(
            text = "Recorded: ${
                SimpleDateFormat(
                    "dd-MM-yyyy HH:mm:ss",
                    Locale.getDefault()
                ).format(Date())
            }",
            fontSize = 11.sp,
            color = Color.Gray
        )

        Spacer(Modifier.height(16.dp))

        /* ---------- ALERT ---------- */

        if (isAbnormal) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = Color(0xFFFFEBEE)
                ),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(
                        text = "⚠ Possible Arrhythmia Detected",
                        color = Color(0xFFC62828),
                        fontSize = 16.sp
                    )

                    Spacer(Modifier.height(8.dp))

                    val context = LocalView.current.context
                    Button(
                        onClick = { viewModel.openEmergencyDialer(context) },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFFC62828)
                        )
                    ) {
                        Text("SOS", color = Color.White)
                    }
                }
            }
        }

        Spacer(Modifier.weight(1f))

        /* ---------- FOOTER ---------- */

        Text(
            text = "For educational and research use only.\nNot for clinical diagnosis.",
            fontSize = 12.sp,
            color = Color.Gray
        )

        Spacer(Modifier.height(12.dp))

        Button(
            onClick = onStatsClick,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("View Arrhythmia Events")
        }
    }
}
