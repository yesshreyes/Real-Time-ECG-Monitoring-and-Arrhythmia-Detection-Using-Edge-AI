package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.rhythmai.presentation.monitor.ArrhythmiaEvent
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ArrhythmiaEventItem(event: ArrhythmiaEvent) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = event.arrhythmiaType,
                    style = MaterialTheme.typography.bodyLarge,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = formatTimestamp(event.timestamp),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
                )
                // NEW: Show quality score
                Text(
                    text = "Quality: ${(event.qualityScore * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (event.qualityScore >= 0.7f) {
                        Color(0xFF4CAF50)
                    } else {
                        Color(0xFFFF9800)
                    }
                )
            }

            // Confidence badge
            Surface(
                shape = RoundedCornerShape(12.dp),
                color = when (event.arrhythmiaType) {
                    "Normal" -> Color(0xFF4CAF50).copy(alpha = 0.2f)
                    "Supraventricular Arrhythmia" -> Color(0xFFFF9800).copy(alpha = 0.2f)
                    "Ventricular Arrhythmia" -> Color(0xFFF44336).copy(alpha = 0.2f)
                    else -> MaterialTheme.colorScheme.surfaceVariant
                }
            ) {
                Text(
                    text = "${(event.confidence * 100).toInt()}%",
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}
private fun formatTimestamp(timestamp: Long): String {
    val sdf = SimpleDateFormat("MMM dd, HH:mm:ss", Locale.getDefault())
    return sdf.format(Date(timestamp))
}