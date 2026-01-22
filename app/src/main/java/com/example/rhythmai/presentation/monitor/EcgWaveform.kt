package com.example.rhythmai.presentation.monitor

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

@Composable
fun EcgWaveform(
    samples: List<Float>,
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(200.dp)
            .background(Color.Black)
    ) {
        val gridSmall = 25f
        val gridBig = gridSmall * 5

        for (x in 0..(size.width / gridSmall).toInt()) {
            drawLine(
                color = Color.DarkGray,
                start = androidx.compose.ui.geometry.Offset(x * gridSmall, 0f),
                end = androidx.compose.ui.geometry.Offset(x * gridSmall, size.height),
                strokeWidth = 1f
            )
        }

        for (y in 0..(size.height / gridSmall).toInt()) {
            drawLine(
                color = Color.DarkGray,
                start = androidx.compose.ui.geometry.Offset(0f, y * gridSmall),
                end = androidx.compose.ui.geometry.Offset(size.width, y * gridSmall),
                strokeWidth = 1f
            )
        }

        for (x in 0..(size.width / gridBig).toInt()) {
            drawLine(
                color = Color.Gray,
                start = androidx.compose.ui.geometry.Offset(x * gridBig, 0f),
                end = androidx.compose.ui.geometry.Offset(x * gridBig, size.height),
                strokeWidth = 2f
            )
        }

        for (y in 0..(size.height / gridBig).toInt()) {
            drawLine(
                color = Color.Gray,
                start = androidx.compose.ui.geometry.Offset(0f, y * gridBig),
                end = androidx.compose.ui.geometry.Offset(size.width, y * gridBig),
                strokeWidth = 2f
            )
        }

        if (samples.size < 2) return@Canvas

        val path = Path()
        val maxAmplitude = 1.5f
        val centerY = size.height / 2
        val xStep = size.width / (samples.size - 1)

        samples.forEachIndexed { index, value ->
            val x = index * xStep
            val y = centerY - (value / maxAmplitude) * centerY

            if (index == 0) path.moveTo(x, y)
            else path.lineTo(x, y)
        }

        drawPath(
            path = path,
            color = Color.Green,
            style = Stroke(width = 3f)
        )
    }
}
