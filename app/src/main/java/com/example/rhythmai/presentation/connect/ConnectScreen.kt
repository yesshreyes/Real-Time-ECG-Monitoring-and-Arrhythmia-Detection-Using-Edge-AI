package com.example.rhythmai.presentation.connect

import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

@Composable
fun ConnectScreen(
    onConnected: () -> Unit,
    viewModel: ConnectViewModel = viewModel()
) {
    val connected by viewModel.isConnected.collectAsState()
    val context = LocalContext.current

    // 🔥 Handle toast + delayed navigation
    LaunchedEffect(connected) {
        if (connected) {
            Toast.makeText(
                context,
                "Device connected successfully",
                Toast.LENGTH_SHORT
            ).show()

            delay(2000) // ⏳ 2 seconds
            onConnected()
        }
    }

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {

            Box(
                modifier = Modifier
                    .size(180.dp)
                    .clip(CircleShape)
                    .background(if (connected) Color.Green else Color.Blue)
                    .clickable(enabled = !connected) {
                        viewModel.connect()
                    },
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = if (connected) "Connected" else "Tap to Connect",
                    color = Color.White
                )
            }
        }
    }
}
