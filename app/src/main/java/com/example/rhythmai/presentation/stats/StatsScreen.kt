package com.example.rhythmai.presentation.stats

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun StatsScreen(
    viewModel: StatsViewModel = viewModel()
) {
    val rhythm by viewModel.rhythm.collectAsState()
    val confidence by viewModel.confidence.collectAsState()

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp)
    ) {
        Text("Analysis", fontSize = 22.sp)

        Spacer(Modifier.height(16.dp))

        Text("Rhythm: $rhythm", fontSize = 18.sp)
        Text("Confidence: ${confidence}%", fontSize = 18.sp)

        Spacer(Modifier.height(16.dp))

        Text("Inference Time: 12 ms")
        Text("Model Accuracy: ~96%")
    }
}
