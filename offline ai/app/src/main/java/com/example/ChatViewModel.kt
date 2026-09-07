package com.example

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/**
 * Modern ViewModel managing multi-turn chat conversations, multi-model switching,
 * real-time token streaming, and on-device GGUF downloads.
 */
class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val gptEngine = BengaliGptEngine()

    private val _isModelReady = MutableStateFlow(false)
    val isModelReady: StateFlow<Boolean> = _isModelReady.asStateFlow()

    private val _activePreset = MutableStateFlow(ModelDownloader.DEFAULT_PRESET)
    val activePreset: StateFlow<ModelDownloader.ModelPreset> = _activePreset.asStateFlow()

    private val _installedPresets = MutableStateFlow<List<ModelDownloader.ModelPreset>>(emptyList())
    val installedPresets: StateFlow<List<ModelDownloader.ModelPreset>> = _installedPresets.asStateFlow()

    private val _downloadState = MutableStateFlow<ModelDownloader.DownloadState>(ModelDownloader.DownloadState.Idle)
    val downloadState: StateFlow<ModelDownloader.DownloadState> = _downloadState.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _showDiagnosticsDialog = MutableStateFlow(false)
    val showDiagnosticsDialog: StateFlow<Boolean> = _showDiagnosticsDialog.asStateFlow()

    private val _showModelInfoDialog = MutableStateFlow(false)
    val showModelInfoDialog: StateFlow<Boolean> = _showModelInfoDialog.asStateFlow()

    private val _showModelSwitcherDialog = MutableStateFlow(false)
    val showModelSwitcherDialog: StateFlow<Boolean> = _showModelSwitcherDialog.asStateFlow()

    private var downloadJob: Job? = null
    private var generationJob: Job? = null

    init {
        checkAndLoadInitialModel()
    }

    fun refreshInstalledPresets() {
        val context = getApplication<Application>()
        _installedPresets.value = ModelDownloader.getInstalledPresets(context)
    }

    /**
     * Checks if any preset or legacy model is already present and loads it into memory.
     */
    fun checkAndLoadInitialModel() {
        val context = getApplication<Application>()
        ModelDownloader.ensureModelMigration(context)
        refreshInstalledPresets()

        val installed = _installedPresets.value
        if (installed.isNotEmpty()) {
            val presetToLoad = installed[0]
            _activePreset.value = presetToLoad
            val file = ModelDownloader.getModelFile(context, presetToLoad)
            loadEngine(file, presetToLoad)
        } else {
            _isModelReady.value = false
            _downloadState.value = ModelDownloader.DownloadState.Idle
        }
    }

    /**
     * Loads a given model preset into memory via native llama.cpp.
     */
    fun loadEngine(modelFile: File, preset: ModelDownloader.ModelPreset) {
        viewModelScope.launch {
            val result = gptEngine.loadModel(modelFile)
            if (result.isSuccess) {
                _activePreset.value = preset
                _isModelReady.value = true
                _errorMessage.value = null
                refreshInstalledPresets()

                if (_messages.value.isEmpty()) {
                    _messages.value = listOf(
                        ChatMessage(
                            text = "স্বাগতম! আমি আপনার অন-ডিভাইস বাংলা এআই সহায়ক। বর্তমানে '${preset.name}' মডেলটি ডিভাইস মেমরিতে সক্রিয় রয়েছে। আপনি যেকোনো প্রশ্ন করতে পারেন।",
                            isUser = false
                        )
                    )
                }
            } else {
                _isModelReady.value = false
                val err = result.exceptionOrNull()?.localizedMessage ?: "মডেল লোড করতে ব্যর্থ হয়েছে।"
                _errorMessage.value = err
            }
        }
    }

    /**
     * Switches the active model to another downloaded preset without re-downloading.
     */
    fun switchModel(preset: ModelDownloader.ModelPreset) {
        val context = getApplication<Application>()
        val file = ModelDownloader.getModelFile(context, preset)
        if (file.exists() && file.length() > 1_000_000L) {
            gptEngine.close()
            loadEngine(file, preset)
            _messages.value = _messages.value + ChatMessage(
                text = "🔄 মডেল পরিবর্তন করা হয়েছে: '${preset.name}' এখন সক্রিয়।",
                isUser = false
            )
        } else {
            _errorMessage.value = "'${preset.name}' ফাইলটি এখনো ডাউনলোড করা হয়নি।"
        }
    }

    /**
     * Begins downloading the specified GGUF model preset.
     */
    fun startModelDownload(preset: ModelDownloader.ModelPreset, customUrl: String? = null) {
        downloadJob?.cancel()
        val context = getApplication<Application>()

        downloadJob = viewModelScope.launch {
            ModelDownloader.downloadModel(context, preset, customUrl).collect { state ->
                _downloadState.value = state
                if (state is ModelDownloader.DownloadState.Success) {
                    refreshInstalledPresets()
                    loadEngine(state.modelFile, state.preset)
                } else if (state is ModelDownloader.DownloadState.Error) {
                    _errorMessage.value = state.message
                }
            }
        }
    }

    /**
     * Imports a local .gguf file from device storage for a specific preset slot.
     */
    fun importModelFromUri(uri: Uri, targetPreset: ModelDownloader.ModelPreset = _activePreset.value) {
        viewModelScope.launch(Dispatchers.IO) {
            val context = getApplication<Application>()
            try {
                _downloadState.value = ModelDownloader.DownloadState.Downloading(
                    bytesRead = 0,
                    totalBytes = targetPreset.sizeMb * 1024 * 1024,
                    percentage = 0,
                    speedBytesPerSec = 0,
                    speedFormatted = "-",
                    downloadedFormatted = "কপি করা হচ্ছে...",
                    statusBengali = "ডিভাইস থেকে ফাইল লোড হচ্ছে..."
                )

                val targetFile = ModelDownloader.getModelFile(context, targetPreset)
                context.contentResolver.openInputStream(uri)?.use { input ->
                    FileOutputStream(targetFile).use { output ->
                        input.copyTo(output)
                    }
                }

                if (targetFile.exists() && targetFile.length() > 1_000_000L) {
                    refreshInstalledPresets()
                    withContext(Dispatchers.Main) {
                        _downloadState.value = ModelDownloader.DownloadState.Success(targetFile, targetPreset)
                        loadEngine(targetFile, targetPreset)
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        _errorMessage.value = "ফাইলটি কপি করতে ব্যর্থ হয়েছে বা ফাইলটি অবৈধ।"
                        _downloadState.value = ModelDownloader.DownloadState.Idle
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    _errorMessage.value = "ফাইল ইমপোর্ট ত্রুটি: ${e.localizedMessage}"
                    _downloadState.value = ModelDownloader.DownloadState.Idle
                }
            }
        }
    }

    fun cancelModelDownload() {
        downloadJob?.cancel()
        val context = getApplication<Application>()
        ModelDownloader.cancelDownload(context)
        _downloadState.value = ModelDownloader.DownloadState.Idle
    }

    /**
     * Sends user query with multi-turn conversation context and streams disciplined response.
     */
    fun sendMessage(userText: String) {
        if (userText.isBlank() || _isGenerating.value || !_isModelReady.value) return

        val trimmedText = userText.trim()
        val userMessage = ChatMessage(text = trimmedText, isUser = true)
        val initialAiMessage = ChatMessage(text = "", isUser = false)
        val aiMessageId = initialAiMessage.id

        val updatedMessages = _messages.value + userMessage
        _messages.value = updatedMessages + initialAiMessage
        _isGenerating.value = true
        _errorMessage.value = null

        generationJob = viewModelScope.launch {
            val responseBuilder = StringBuilder()
            try {
                // Pass full conversation history for context adherence
                gptEngine.generateStream(updatedMessages).collect { tokenPiece ->
                    responseBuilder.append(tokenPiece)
                    val cleaned = gptEngine.cleanResponse(responseBuilder.toString())
                    _messages.value = _messages.value.map { msg ->
                        if (msg.id == aiMessageId) {
                            msg.copy(text = cleaned)
                        } else {
                            msg
                        }
                    }
                }

                val currentText = _messages.value.find { it.id == aiMessageId }?.text.orEmpty()
                if (currentText.isBlank()) {
                    _messages.value = _messages.value.map { msg ->
                        if (msg.id == aiMessageId) {
                            msg.copy(text = "(মডেল কোনো উত্তর তৈরি করেনি।)")
                        } else {
                            msg
                        }
                    }
                }
            } catch (e: Throwable) {
                val errorText = "ইনফারেন্স ত্রুটি: ${e.localizedMessage ?: "অনাকাঙ্ক্ষিত ত্রুটি"}"
                _errorMessage.value = errorText
                _messages.value = _messages.value.map { msg ->
                    if (msg.id == aiMessageId) {
                        msg.copy(text = "❌ $errorText")
                    } else {
                        msg
                    }
                }
            } finally {
                _isGenerating.value = false
            }
        }
    }

    fun clearChat() {
        _messages.value = emptyList()
    }

    fun dismissError() {
        _errorMessage.value = null
    }

    fun setShowDiagnostics(show: Boolean) {
        _showDiagnosticsDialog.value = show
    }

    fun setShowModelInfo(show: Boolean) {
        _showModelInfoDialog.value = show
    }

    fun setShowModelSwitcher(show: Boolean) {
        _showModelSwitcherDialog.value = show
        if (show) {
            refreshInstalledPresets()
        }
    }

    fun deletePreset(preset: ModelDownloader.ModelPreset) {
        val context = getApplication<Application>()
        if (_activePreset.value.id == preset.id) {
            gptEngine.close()
            _isModelReady.value = false
        }
        ModelDownloader.deletePreset(context, preset)
        refreshInstalledPresets()
    }

    override fun onCleared() {
        super.onCleared()
        downloadJob?.cancel()
        generationJob?.cancel()
        gptEngine.close()
    }
}
