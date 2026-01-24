package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
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
fun ArrhythmiaEventCard(event: ArrhythmiaEvent) {
    val time = remember(event.timestamp) {
        SimpleDateFormat("dd MMM, HH:mm:ss", Locale.getDefault())
            .format(Date(event.timestamp))
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFF8F8F8))
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = event.arrhythmiaType,
                fontWeight = FontWeight.Bold
            )
            Text("Confidence: ${(event.confidence * 100).toInt()}%")
            Text(
                text = time,
                fontSize = 12.sp,
                color = Color.Gray
            )
        }
    }
}
