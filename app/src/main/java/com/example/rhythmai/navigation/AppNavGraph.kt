package com.example.rhythmai.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.rhythmai.presentation.connect.ConnectScreen
import com.example.rhythmai.presentation.monitor.MonitorScreen
import com.example.rhythmai.presentation.stats.StatsScreen

@Composable
fun AppNavGraph() {
    val navController = rememberNavController()

    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    val showBottomBar = currentRoute != "connect"

    Scaffold(
        bottomBar = {
            if (showBottomBar) {
                BottomBar(navController)
            }
        }
    ) { padding ->

        NavHost(
            navController = navController,
            startDestination = "connect",
            modifier = Modifier.padding(padding)
        ) {

            composable("connect") {
                ConnectScreen(
                    onConnected = {
                        navController.navigate("monitor") {
                            popUpTo("connect") { inclusive = true }
                        }
                    }
                )
            }

            composable("monitor") {
                MonitorScreen(
                    onStatsClick = {
                        navController.navigate("stats")
                    }
                )
            }

            composable("stats") {
                StatsScreen()
            }
        }
    }
}
