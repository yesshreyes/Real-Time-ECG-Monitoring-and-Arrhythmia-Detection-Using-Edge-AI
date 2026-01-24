package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun StatsScreen(
    viewModel: StatsViewModel
) {
    val events by viewModel.events.collectAsState()

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {

        /* ---------------- MODEL ARCHITECTURE ---------------- */

        item {
            SectionTitle("Model Architecture")
            viewModel.modelArchitecture.forEach {
                StatRow(it.first, it.second)
            }
            SmallNote(
                "Stage 1 filters normal beats quickly. Stage 2 classifies abnormal beats precisely."
            )
        }

        item { Divider() }

        /* ---------------- MODEL PERFORMANCE ---------------- */

        item {
            SectionTitle("Model Performance (Offline Evaluation)")
            viewModel.modelPerformance.forEach {
                StatRow(it.first, it.second)
            }
            SmallNote(
                "Metrics obtained during offline evaluation using labeled ECG data."
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
                    color = Color.Gray,
                    fontSize = 14.sp
                )
            }
        } else {
            items(events.reversed()) { event ->
                ArrhythmiaEventCard(event)
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