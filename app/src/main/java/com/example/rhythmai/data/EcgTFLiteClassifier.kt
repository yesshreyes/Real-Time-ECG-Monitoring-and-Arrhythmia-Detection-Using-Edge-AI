package com.example.rhythmai.data

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class EcgTFLiteClassifier(context: Context) {

    private val interpreter: Interpreter

    init {
        val assetList = context.assets.list("")?.joinToString(", ")
        Log.e("TFLite_DEBUG", "ASSETS: $assetList")

        val modelName = "ecg_multiclass_compat.tflite"
        Log.e("TFLite_DEBUG", "LOADING MODEL: $modelName")

        val modelBytes = context.assets.open(modelName).readBytes()
        val buffer = ByteBuffer.allocateDirect(modelBytes.size)
            .order(ByteOrder.nativeOrder())
        buffer.put(modelBytes)

        interpreter = Interpreter(buffer)
    }


    fun predict(window: FloatArray): Pair<String, Float> {

        // Input shape: [1, 360, 1]
        val input = Array(1) {
            Array(window.size) { floatArrayOf(0f) }
        }

        for (i in window.indices) {
            input[0][i][0] = window[i]
        }

        // Output shape: [1, 3]
        val output = Array(1) { FloatArray(3) }

        interpreter.run(input, output)

        val probs = output[0]

        val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
        val confidence = probs[maxIndex]

        val label = when (maxIndex) {
            0 -> "Normal"
            1 -> "Supraventricular Arrhythmia"
            2 -> "Ventricular Arrhythmia"
            else -> "Unknown"
        }

        return label to confidence
    }
}
