package com.example.rhythmai.data

import android.content.Context
import java.io.BufferedReader
import java.io.InputStreamReader

object EcgAssetLoader {

    fun loadEcg(
        context: Context,
        fileName: String = "100_ekg.csv",
        leadName: String = "MLII"
    ): List<Float> {

        val samples = mutableListOf<Float>()

        val reader = BufferedReader(
            InputStreamReader(context.assets.open(fileName))
        )

        // Read header
        val header = reader.readLine().split(",")
        val leadIndex = header.indexOf(leadName)

        require(leadIndex != -1) {
            "ECG lead $leadName not found in $fileName"
        }

        reader.forEachLine { line ->
            val cols = line.split(",")

            cols.getOrNull(leadIndex)
                ?.toFloatOrNull()
                ?.let { samples.add(it) }
        }

        reader.close()
        return samples
    }
}
