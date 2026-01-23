package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.rhythmai.presentation.monitor.MonitorViewModel
import java.text.SimpleDateFormat
import java.util.*

@Composable
fun StatsScreen(
    viewModel: MonitorViewModel
) {

    val events by viewModel.events.collectAsState()

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        /* ---------------- MODEL PERFORMANCE ---------------- */

        item {
            SectionTitle("Model Performance (Offline Evaluation)")

            StatRow("Dataset", "MIT-BIH Arrhythmia")
            StatRow("Classes", "Normal, SV, Ventricular")
            StatRow("Validation Accuracy", "94.6%")
            StatRow("Evaluation Method", "Inter-patient split")

            SmallNote(
                "Metrics obtained during offline evaluation using labeled ECG data."
            )
        }

        item { Divider() }

        /* ---------------- ON-DEVICE INFERENCE ---------------- */

        item {
            SectionTitle("On-device Inference")

            StatRow("Execution", "On-device (TensorFlow Lite)")
            StatRow("Model Size", "≈ 420 KB")
            StatRow("Avg Inference Time", "10–15 ms")
            StatRow("Confidence Threshold", "0.5")

            SmallNote(
                "Real-time inference performed entirely on the device without cloud dependency."
            )
        }

        item { Divider() }

        /* ---------------- ARRHYTHMIA EVENTS ---------------- */

        item {
            SectionTitle("Detected Arrhythmia Events")
        }

        if (events.isEmpty()) {
            item {
                Text(
                    text = "No arrhythmia events detected yet.",
                    fontSize = 14.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else {
            items(events.reversed()) { event ->
                ArrhythmiaEventItem(event)
            }
        }
    }
}

/* ---------------- UI HELPERS ---------------- */

@Composable
private fun SectionTitle(title: String) {
    Text(
        text = title,
        fontSize = 20.sp,
        color = MaterialTheme.colorScheme.primary
    )
}

@Composable
private fun StatRow(label: String, value: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label, fontSize = 15.sp)
        Text(value, fontSize = 15.sp)
    }
}

@Composable
private fun SmallNote(text: String) {
    Spacer(Modifier.height(4.dp))
    Text(
        text = text,
        fontSize = 12.sp,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}

@Composable
private fun ArrhythmiaEventItem(event: com.example.rhythmai.presentation.monitor.ArrhythmiaEvent) {

    val time = remember(event.timestamp) {
        SimpleDateFormat("HH:mm:ss", Locale.getDefault())
            .format(Date(event.timestamp))
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text("Time: $time", fontSize = 14.sp)
        Text(
            text = "Confidence: ${(event.confidence * 100).toInt()}%",
            fontSize = 14.sp
        )
    }
}
