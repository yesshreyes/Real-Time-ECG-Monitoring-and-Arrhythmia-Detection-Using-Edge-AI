package com.example.rhythmai.presentation.connect

import android.widget.Toast
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Bluetooth
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay

private val EcgGreen = Color(0xFF00E676)
private val DeepBlue = Color(0xFF0A1A2F)
private val MutedBlue = Color(0xFF1E3A5F)
private val SoftGray = Color(0xFFB0BEC5)

@Composable
fun ConnectScreen(
    onConnected: () -> Unit,
    viewModel: ConnectViewModel = viewModel()
) {
    val connected by viewModel.isConnected.collectAsState()
    val context = LocalContext.current

    LaunchedEffect(connected) {
        if (connected) {
            Toast.makeText(context, "ECG Device Connected", Toast.LENGTH_SHORT).show()
            delay(600)
            onConnected()
        }
    }

    val infiniteTransition = rememberInfiniteTransition(label = "pulse")

    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseScale"
    )

    val pulseAlpha by infiniteTransition.animateFloat(
        initialValue = 0.3f,
        targetValue = 0.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1200),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseAlpha"
    )

    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(2500, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "bluetoothRotation"
    )


    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(DeepBlue),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {

            Box(contentAlignment = Alignment.Center) {

                // 🔵 Pulse Ring
                if (!connected) {
                    Box(
                        modifier = Modifier
                            .size(220.dp)
                            .scale(pulseScale)
                            .clip(CircleShape)
                            .background(EcgGreen.copy(alpha = pulseAlpha))
                    )
                }

                // 🔘 Main Button
                Box(
                    modifier = Modifier
                        .size(160.dp)
                        .clip(CircleShape)
                        .background(if (connected) EcgGreen else MutedBlue)
                        .clickable(enabled = !connected) {
                            viewModel.connect()
                        },
                    contentAlignment = Alignment.Center
                ) {
                    if (connected) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "Connected",
                            tint = Color.White,
                            modifier = Modifier.size(48.dp)
                        )
                    } else {
                        Icon(
                            imageVector = Icons.Default.Bluetooth,
                            contentDescription = "Bluetooth",
                            tint = Color.White,
                            modifier = Modifier
                                .size(42.dp)
                                .graphicsLayer {
                                    rotationZ = rotation
                                }
                        )
                    }

                }
            }

            Spacer(Modifier.height(24.dp))

            Text(
                text = if (connected) "Bluetooth Connected"
                else "Searching for ECG device…",
                color = SoftGray,
                fontSize = 14.sp
            )
        }
    }
}
