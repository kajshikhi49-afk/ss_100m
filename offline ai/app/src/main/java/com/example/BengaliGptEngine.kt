package com.example

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Universal Native On-Device GGUF Inference Engine powered by custom-extended llama.cpp.
 *
 * Supports:
 * 1. Our Custom 50M BengaliGPT (GGUF Q8_0 - Native Exact GELU & SuperBPE)
 * 2. SmolLM-135M-Instruct (GGUF Q4_K_M)
 * 3. Qwen2.5-0.5B-Instruct (GGUF Q4_K_M)
 */
class BengaliGptEngine {

    companion object {
        private const val TAG = "BengaliGptEngine"

        init {
            val libs = listOf("omp", "ggml-base", "ggml-cpu", "ggml", "llama", "bengali_ai_jni")
            for (lib in libs) {
                try {
                    System.loadLibrary(lib)
                    Log.i(TAG, "Loaded native library: $lib")
                } catch (t: Throwable) {
                    Log.w(TAG, "System.loadLibrary($lib) notice: ${t.message}")
                }
            }
        }

        const val MAX_NEW_TOKENS = 200
        const val TEMPERATURE = 0.5f
        const val TOP_K = 40
        const val TOP_P = 0.85f
        const val REPETITION_PENALTY = 1.35f

        val STOP_SEQUENCES = listOf(
            "\nপ্রশ্ন:",
            "প্রশ্ন:",
            "<EOS>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "\nuser\n",
            "\nUser:"
        )

        const val STOP_TOKEN = "<EOS>"
    }

    interface TokenCallback {
        /**
         * Called when new token bytes are generated.
         * Return true to continue, false to stop immediately.
         */
        fun onTokenBytes(bytes: ByteArray): Boolean
    }

    var isLoaded: Boolean = false
        private set

    var loadedFilePath: String? = null
        private set

    var lastLoadError: String? = null
        private set

    suspend fun loadModel(modelFile: File): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (!modelFile.exists() || modelFile.length() < 1_000_000L) {
                val err = "মডেল ফাইলটি পাওয়া যায়নি বা ফাইলের আকার অবৈধ (${modelFile.absolutePath})"
                lastLoadError = err
                isLoaded = false
                return@withContext Result.failure(Exception(err))
            }

            Log.i(TAG, "Loading GGUF model via native llama.cpp: ${modelFile.absolutePath} (${modelFile.length() / (1024 * 1024)} MB)")

            val nThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
            val success = nativeLoadModel(modelFile.absolutePath, nThreads)

            if (success) {
                isLoaded = true
                loadedFilePath = modelFile.absolutePath
                lastLoadError = null
                Log.i(TAG, "GGUF Model loaded successfully: ${modelFile.name}")
                Result.success(Unit)
            } else {
                isLoaded = false
                loadedFilePath = null
                val err = "নেটিভ llama.cpp মডেলে লোড ব্যর্থ হয়েছে।"
                lastLoadError = err
                Log.e(TAG, err)
                Result.failure(Exception(err))
            }
        } catch (e: Exception) {
            isLoaded = false
            loadedFilePath = null
            lastLoadError = e.message
            Log.e(TAG, "Exception during nativeLoadModel: ${e.message}", e)
            Result.failure(e)
        }
    }

    /**
     * Streams generated tokens via Kotlin Flow using native llama.cpp sampling.
     * Enforces strict multi-token stop word detection to prevent endless generation.
     */
    fun generateStream(messages: List<ChatMessage>): Flow<String> = callbackFlow {
        val modelPath = loadedFilePath ?: ""
        val formattedPrompt = formatMultiTurnPrompt(messages, modelPath)

        val accumulated = StringBuilder()
        var hasStopped = false

        val callback = object : TokenCallback {
            override fun onTokenBytes(bytes: ByteArray): Boolean {
                if (hasStopped) return false

                val piece = String(bytes, Charsets.UTF_8)
                accumulated.append(piece)
                val currentText = accumulated.toString()

                for (stop in STOP_SEQUENCES) {
                    if (currentText.contains(stop)) {
                        hasStopped = true
                        Log.i(TAG, "Stop sequence '$stop' matched in stream. Halting C++ generator.")
                        return false
                    }
                }

                val sendResult = trySend(piece)
                return sendResult.isSuccess
            }
        }

        try {
            nativeGenerate(
                prompt = formattedPrompt,
                maxTokens = MAX_NEW_TOKENS,
                temperature = TEMPERATURE,
                topK = TOP_K,
                repetitionPenalty = REPETITION_PENALTY,
                callback = callback
            )
        } catch (e: Exception) {
            Log.e(TAG, "Generation error: ${e.message}", e)
        } finally {
            channel.close()
        }

        awaitClose {
            nativeCancelGeneration()
        }
    }.flowOn(Dispatchers.Default)

    /**
     * Formats conversation history into a structured prompt matching the model architecture.
     */
    fun formatMultiTurnPrompt(messages: List<ChatMessage>, modelPath: String): String {
        val lower = modelPath.lowercase()
        val isChatML = lower.contains("smollm") || lower.contains("qwen")

        // Take last 4 conversational messages to keep prompt concise for small context
        val recentMessages = messages.takeLast(4)

        return if (isChatML) {
            val sb = StringBuilder()
            sb.append("<|im_start|>system\nYou are a helpful and disciplined offline AI assistant. Always respond concisely in Bengali.<|im_end|>\n")
            for (msg in recentMessages) {
                val role = if (msg.isUser) "user" else "assistant"
                if (msg.text.isNotBlank()) {
                    sb.append("<|im_start|>$role\n${msg.text.trim()}<|im_end|>\n")
                }
            }
            sb.append("<|im_start|>assistant\n")
            sb.toString()
        } else {
            // BengaliGPT standard training format:
            val sb = StringBuilder()
            for (msg in recentMessages) {
                if (msg.isUser) {
                    sb.append("প্রশ্ন: ${msg.text.trim()} ")
                } else if (msg.text.isNotBlank()) {
                    sb.append("উত্তর: ${msg.text.trim()} <EOS>\n")
                }
            }
            sb.append("উত্তর:")
            sb.toString()
        }
    }

    /**
     * Cleans response output string from any residual prompt markers or stop sequences.
     */
    fun cleanResponse(rawText: String): String {
        var text = rawText
        if (text.contains("উত্তর:")) {
            text = text.substringAfter("উত্তর:")
        }
        for (stop in STOP_SEQUENCES) {
            if (text.contains(stop)) {
                text = text.substringBefore(stop)
            }
        }
        return text.trim()
    }

    fun close() {
        nativeClose()
        isLoaded = false
        loadedFilePath = null
    }

    // Native JNI Declarations
    private external fun nativeLoadModel(modelPath: String, nThreads: Int): Boolean
    private external fun nativeCancelGeneration()
    private external fun nativeGenerate(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        repetitionPenalty: Float,
        callback: TokenCallback
    )
    private external fun nativeClose()
}
