package com.example.rhythmai.presentation.monitor

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.rhythmai.data.QualityStatus
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun PredictionCard(
    prediction: String,
    confidence: Float,
    qualityStatus: QualityStatus,
    bpm: Int,
    onCaptureClick: () -> Unit
) {
    val recordedTime = rememberLiveTime()

    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {

            /* ---------- HEART RATE + CAPTURE BUTTON ---------- */

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.weight(1f)
                ) {
                    Icon(
                        imageVector = Icons.Default.Favorite,
                        contentDescription = "Heart Rate",
                        tint = Color(0xFFE53935),
                        modifier = Modifier.size(32.dp)
                    )

                    Spacer(Modifier.width(8.dp))

                    Text(
                        text = "$bpm",
                        fontSize = 36.sp,   // ❤️ BIG
                        fontWeight = FontWeight.Bold
                    )

                    Spacer(Modifier.width(4.dp))

                    Text(
                        text = "BPM",
                        fontSize = 16.sp,
                        color = Color.Gray,
                        modifier = Modifier.padding(top = 8.dp)
                    )
                }

                IconButton(onClick = onCaptureClick) {
                    Icon(
                        imageVector = Icons.Default.CameraAlt,
                        contentDescription = "Capture ECG",
                        tint = Color(0xFF2E7D32),
                        modifier = Modifier.size(28.dp)
                    )
                }
            }

            Divider()

            /* ---------- PREDICTION ---------- */

            Text(
                text = prediction,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = when (prediction) {
                    "Normal" -> Color(0xFF4CAF50)
                    "Supraventricular Arrhythmia" -> Color(0xFFFF9800)
                    "Ventricular Arrhythmia" -> Color(0xFFF44336)
                    else -> MaterialTheme.colorScheme.onSurface
                }
            )

            /* ---------- CONFIDENCE ---------- */

            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {

                LinearProgressIndicator(
                    progress = confidence,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(8.dp)
                        .clip(RoundedCornerShape(4.dp)),
                    color = when (prediction) {
                        "Normal" -> Color(0xFF4CAF50)
                        "Supraventricular Arrhythmia" -> Color(0xFFFF9800)
                        "Ventricular Arrhythmia" -> Color(0xFFF44336)
                        else -> MaterialTheme.colorScheme.primary
                    }
                )

                Text(
                    text = "Confidence ${(confidence * 100).toInt()}%",
                    fontSize = 12.sp,
                    color = Color.Gray
                )

                Text(
                    text = "Recorded: $recordedTime",
                    fontSize = 11.sp,
                    color = Color.Gray
                )
            }


            /* ---------- QUALITY WARNING ---------- */

            if (qualityStatus == QualityStatus.BAD) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Warning,
                        contentDescription = "Low quality",
                        tint = Color(0xFFFF9800),
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(Modifier.width(6.dp))
                    Text(
                        text = "Poor signal quality detected",
                        fontSize = 12.sp,
                        color = Color(0xFFE65100)
                    )
                }
            }
        }
    }
}

@Composable
private fun rememberLiveTime(): String {
    var time by remember { mutableStateOf(System.currentTimeMillis()) }

    LaunchedEffect(Unit) {
        while (true) {
            time = System.currentTimeMillis()
            kotlinx.coroutines.delay(1000)
        }
    }

    return SimpleDateFormat(
        "dd MMM yyyy, HH:mm:ss",
        Locale.getDefault()
    ).format(Date(time))
}
