package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.rhythmai.data.PredictionResult
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

        /* ---------------- MODEL ARCHITECTURE ---------------- */

        item {
            SectionTitle("Model Architecture")
            StatRow("Type", "Two-Stage Cascade")
            StatRow("Stage 1", "Normal vs Abnormal (Screening)")
            StatRow("Stage 2", "SV vs Ventricular (Diagnosis)")
            StatRow("Total Model Size", "≈ 1 MB")
            SmallNote(
                "Stage 1 filters normal beats quickly. Stage 2 classifies abnormal beats precisely."
            )
        }

        item { Divider() }

        /* ---------------- MODEL PERFORMANCE ---------------- */

        item {
            SectionTitle("Model Performance (Offline Evaluation)")
            StatRow("Dataset", "MIT-BIH Arrhythmia")
            StatRow("Stage 1 Sensitivity", "≥ 95%")
            StatRow("Stage 2 F1-Score", "≥ 90%")
            StatRow("Validation Method", "Inter-patient split")
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
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = Color(0xFFF8F8F8)
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        ArrhythmiaEventItem(event)
                    }
                }
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
private fun ArrhythmiaEventItem(
    event: com.example.rhythmai.presentation.monitor.ArrhythmiaEvent
) {
    val time = remember(event.timestamp) {
        SimpleDateFormat("HH:mm:ss", Locale.getDefault())
            .format(Date(event.timestamp))
    }

    val (labelColor, bgColor) = when (event.arrhythmiaType) {
        "Supraventricular Arrhythmia" -> {
            Color(0xFFF9A825) to Color(0xFFFFF8E1)
        }
        "Ventricular Arrhythmia" -> {
            Color(0xFFC62828) to Color(0xFFFFEBEE)
        }
        else -> {
            Color.Gray to Color(0xFFF5F5F5)
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                color = bgColor,
                shape = MaterialTheme.shapes.medium
            )
            .padding(12.dp)
    ) {

        // Time
        Text(
            text = time,
            fontSize = 12.sp,
            color = Color.Gray
        )

        Spacer(Modifier.height(4.dp))

        // Arrhythmia type
        Text(
            text = event.arrhythmiaType,
            fontSize = 15.sp,
            fontWeight = FontWeight.SemiBold,
            color = labelColor
        )

        Spacer(Modifier.height(2.dp))

        // Confidence
        Text(
            text = "Confidence ${(event.confidence * 100).toInt()}%",
            fontSize = 13.sp,
            color = Color.DarkGray
        )
    }
}


@Composable
fun DetailedPredictionView(
    prediction: PredictionResult,
    modifier: Modifier = Modifier
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFF5F5F5)
        )
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Detailed Analysis",
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
            // Stage 1
            Text(
                text = "Stage 1 (Screening)",
                fontSize = 14.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = "Abnormal probability: ${(prediction.stage1AbnormalProb * 100).toInt()}%",
                fontSize = 13.sp
            )
            // Stage 2 (if applicable)
            if (prediction.stage2SvProb != null && prediction.stage2VentProb != null) {
                Divider()

                Text(
                    text = "Stage 2 (Diagnosis)",
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    text = "SV: ${(prediction.stage2SvProb * 100).toInt()}%",
                    fontSize = 13.sp
                )
                Text(
                    text = "Ventricular: ${(prediction.stage2VentProb * 100).toInt()}%",
                    fontSize = 13.sp
                )
            }
            Divider()
            Text(
                text = "Final: ${prediction.finalClass}",
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                color = when (prediction.finalClass) {
                    "Normal" -> Color(0xFF2E7D32)
                    "Supraventricular Arrhythmia" -> Color(0xFFF9A825)
                    "Ventricular Arrhythmia" -> Color(0xFFC62828)
                    else -> Color.Gray
                }
            )
        }
    }
}