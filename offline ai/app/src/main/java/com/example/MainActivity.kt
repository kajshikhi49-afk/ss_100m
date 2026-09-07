package com.example

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.example.ui.theme.MyApplicationTheme

/**
 * Main application entry point for the Bengali Offline AI chatbot.
 */
class MainActivity : ComponentActivity() {

    private val viewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            MyApplicationTheme {
                val isModelReady by viewModel.isModelReady.collectAsState()
                val activePreset by viewModel.activePreset.collectAsState()
                val installedPresets by viewModel.installedPresets.collectAsState()
                val downloadState by viewModel.downloadState.collectAsState()
                val messages by viewModel.messages.collectAsState()
                val isGenerating by viewModel.isGenerating.collectAsState()
                val errorMessage by viewModel.errorMessage.collectAsState()
                val showDiagnosticsDialog by viewModel.showDiagnosticsDialog.collectAsState()
                val showModelInfoDialog by viewModel.showModelInfoDialog.collectAsState()
                val showModelSwitcherDialog by viewModel.showModelSwitcherDialog.collectAsState()

                if (!isModelReady) {
                    DownloadScreen(
                        downloadState = downloadState,
                        activePreset = activePreset,
                        installedPresets = installedPresets,
                        onStartDownload = { preset, url -> viewModel.startModelDownload(preset, url) },
                        onImportFromUri = { uri, preset -> viewModel.importModelFromUri(uri, preset) },
                        onCancelDownload = { viewModel.cancelModelDownload() },
                        onShowDiagnostics = { viewModel.setShowDiagnostics(true) },
                        onSwitchModel = { preset -> viewModel.switchModel(preset) }
                    )
                } else {
                    ChatScreen(
                        messages = messages,
                        isGenerating = isGenerating,
                        errorMessage = errorMessage,
                        showDiagnosticsDialog = showDiagnosticsDialog,
                        showModelInfoDialog = showModelInfoDialog,
                        showModelSwitcherDialog = showModelSwitcherDialog,
                        activePreset = activePreset,
                        installedPresets = installedPresets,
                        onSendMessage = { query -> viewModel.sendMessage(query) },
                        onClearChat = { viewModel.clearChat() },
                        onDismissError = { viewModel.dismissError() },
                        onToggleDiagnostics = { show -> viewModel.setShowDiagnostics(show) },
                        onToggleModelInfo = { show -> viewModel.setShowModelInfo(show) },
                        onToggleModelSwitcher = { show -> viewModel.setShowModelSwitcher(show) },
                        onSwitchModel = { preset -> viewModel.switchModel(preset) },
                        onDownloadPreset = { preset -> viewModel.startModelDownload(preset) },
                        onDeletePreset = { preset -> viewModel.deletePreset(preset) },
                        onImportModel = { uri -> viewModel.importModelFromUri(uri, viewModel.activePreset.value) }
                    )
                }
            }
        }
    }
}

/**
 * Preserved for screenshot test compatibility.
 */
@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(text = "Hello $name!", modifier = modifier)
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    MyApplicationTheme { Greeting("Android") }
}
