package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.rhythmai.data.PredictionResult

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