package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ArrhythmiaEventItem(
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
