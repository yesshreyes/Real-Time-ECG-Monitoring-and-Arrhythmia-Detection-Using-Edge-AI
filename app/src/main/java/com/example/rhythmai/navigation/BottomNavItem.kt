package com.example.rhythmai.navigation

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.ShowChart
import androidx.compose.ui.graphics.vector.ImageVector

sealed class BottomNavItem(
    val route: String,
    val label: String,
    val icon: ImageVector
) {
    object Monitor : BottomNavItem(
        route = "monitor",
        label = "Monitor",
        icon = Icons.Default.Favorite
    )

    object Stats : BottomNavItem(
        route = "stats",
        label = "Stats",
        icon = Icons.Default.ShowChart
    )
}
