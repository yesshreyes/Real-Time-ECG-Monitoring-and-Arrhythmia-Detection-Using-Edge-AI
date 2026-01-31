package com.example.rhythmai.presentation.stats
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.text.SimpleDateFormat
import java.util.*
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
        fontWeight = FontWeight.Bold,
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
        Text(value, fontSize = 15.sp, fontWeight = FontWeight.Medium)
    }
}
@Composable
private fun SmallNote(text: String) {
    Spacer(Modifier.height(4.dp))
    Text(
        text = text,
        fontSize = 12.sp,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        lineHeight = 16.sp
    )
}
private fun formatTimestamp(timestamp: Long): String {
    val sdf = SimpleDateFormat("MMM dd, HH:mm:ss", Locale.getDefault())
    return sdf.format(Date(timestamp))
}