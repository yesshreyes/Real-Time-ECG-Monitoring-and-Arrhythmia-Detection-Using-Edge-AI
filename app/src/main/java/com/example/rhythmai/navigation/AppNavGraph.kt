package com.example.rhythmai.navigation

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.rhythmai.presentation.connect.ConnectScreen
import com.example.rhythmai.presentation.monitor.MonitorScreen
import com.example.rhythmai.presentation.monitor.MonitorViewModel
import com.example.rhythmai.presentation.monitor.MonitorViewModelFactory
import com.example.rhythmai.presentation.stats.StatsScreen
import com.example.rhythmai.presentation.stats.StatsViewModel

@Composable
fun AppNavGraph() {
    val navController = rememberNavController()

    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route

    val showBottomBar = currentRoute != "connect"

    val context = androidx.compose.ui.platform.LocalContext.current

    val monitorViewModel: MonitorViewModel = viewModel(
        factory = MonitorViewModelFactory(context)
    )

    val statsViewModel: StatsViewModel = remember {
        StatsViewModel(monitorViewModel)
    }

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
                    viewModel = monitorViewModel,
                    onStatsClick = { navController.navigate("stats") }
                )
            }

            composable("stats") {
                StatsScreen(
                    viewModel = statsViewModel
                )
            }
        }
    }
}
