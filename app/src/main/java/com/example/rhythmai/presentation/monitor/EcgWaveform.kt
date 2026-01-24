package com.example.rhythmai.presentation.monitor

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp

@Composable
fun EcgWaveform(
    samples: List<Float>,
    samplingRate: Int = 360, // Hz
    modifier: Modifier = Modifier
) {
    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .height(220.dp)
            .background(Color(0xFFFFFDFB)) // ECG paper
    ) {

        /* ---------- Physical calibration ---------- */

        val dpi = density * 160f
        val mmPerPx = 25.4f / dpi

        val smallBoxPx = 1f / mmPerPx   // 1 mm
        val bigBoxPx = 5f / mmPerPx     // 5 mm

        // Time scale: 25 mm/s
        val pxPerSecond = 25f * smallBoxPx
        val pxPerSample = pxPerSecond / samplingRate

        // Voltage scale: 10 mm = 1 mV
        val pxPerMv = 10f * smallBoxPx

        val centerY = size.height / 2

        /* ---------- ECG grid (FIXED) ---------- */

        val smallGrid = Color(0xFFFFCDD2)
        val bigGrid = Color(0xFFE57373)

        // Vertical grid
        var x = 0f
        var col = 0
        while (x <= size.width) {
            drawLine(
                color = if (col % 5 == 0) bigGrid else smallGrid,
                start = Offset(x, 0f),
                end = Offset(x, size.height),
                strokeWidth = if (col % 5 == 0) 2f else 1f
            )
            x += smallBoxPx
            col++
        }

        // Horizontal grid
        var y = 0f
        var row = 0
        while (y <= size.height) {
            drawLine(
                color = if (row % 5 == 0) bigGrid else smallGrid,
                start = Offset(0f, y),
                end = Offset(size.width, y),
                strokeWidth = if (row % 5 == 0) 2f else 1f
            )
            y += smallBoxPx
            row++
        }

        /* ---------- ECG trace (FIXED) ---------- */

        if (samples.size < 2) return@Canvas

        val path = Path()

        // Only draw what fits the screen (CRITICAL FIX)
        val maxVisibleSamples = (size.width / pxPerSample).toInt()
        val visibleSamples = samples.takeLast(maxVisibleSamples)

        visibleSamples.forEachIndexed { index, valueMv ->
            val xPos = index * pxPerSample
            val yPos = centerY - (valueMv * pxPerMv)

            if (index == 0) path.moveTo(xPos, yPos)
            else path.lineTo(xPos, yPos)
        }

        drawPath(
            path = path,
            color = Color.Black,
            style = Stroke(width = 2.2f)
        )
    }
}
