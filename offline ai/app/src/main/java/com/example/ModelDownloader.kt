package com.example

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import okhttp3.Call
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.coroutineContext

/**
 * Robust coroutine-based downloader and manager for multi-model GGUF models.
 * Manages:
 * 1. Our Custom Bengali GPT 50M
 * 2. SmolLM-135M-Instruct
 * 3. Qwen2.5-0.5B-Instruct
 */
object ModelDownloader {

    data class ModelPreset(
        val id: String,
        val name: String,
        val shortName: String,
        val description: String,
        val fileName: String,
        val url: String,
        val sizeMb: Long,
        val isArm32Optimized: Boolean
    )

    val PRESETS = listOf(
        ModelPreset(
            id = "custom_50m",
            name = "আমাদের নিজস্ব বাংলা GPT (50M Q8_0)",
            shortName = "বাংলা GPT 50M",
            description = "Exact GELU + SuperBPE সহ শতভাগ বাংলা অপ্টিমাইজড (~86 MB)",
            fileName = "bengali_gpt_50m_q8_0.gguf",
            url = "https://huggingface.co/kajshikhi/bengali-gpt-50m-gguf/resolve/main/bengali_gpt_50m_q8_0.gguf",
            sizeMb = 86L,
            isArm32Optimized = true
        ),
        ModelPreset(
            id = "qwen_05b",
            name = "Qwen2.5-0.5B-Instruct (ModelScope CDN)",
            shortName = "Qwen 0.5B",
            description = "উন্নত বহুভাষিক ও বাংলা জ্ঞানসম্পন্ন হাই-স্পিড CDN (~468 MB)",
            fileName = "qwen2.5_0.5b_instruct_q4_k_m.gguf",
            url = "https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/master/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            sizeMb = 468L,
            isArm32Optimized = true
        ),
        ModelPreset(
            id = "smollm_135m",
            name = "SmolLM-135M-Instruct (ModelScope CDN)",
            shortName = "SmolLM 135M",
            description = "সুপার-লাইটওয়েট LLaMA আর্কিটেকচার GGUF (~100 MB)",
            fileName = "smollm_135m_instruct_q4_k_m.gguf",
            url = "https://www.modelscope.cn/models/second-state/SmolLM-135M-Instruct-GGUF/resolve/master/SmolLM-135M-Instruct-Q4_K_M.gguf",
            sizeMb = 100L,
            isArm32Optimized = true
        )
    )

    val DEFAULT_PRESET = PRESETS[0]

    const val MODEL_FILE_NAME = "model.gguf"

    private val okHttpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(45, TimeUnit.SECONDS)
            .readTimeout(90, TimeUnit.SECONDS)
            .followRedirects(true)
            .followSslRedirects(true)
            .build()
    }

    private val currentCall = AtomicReference<Call?>(null)

    sealed interface DownloadState {
        data object Idle : DownloadState

        data class Downloading(
            val bytesRead: Long,
            val totalBytes: Long,
            val percentage: Int,
            val speedBytesPerSec: Long,
            val speedFormatted: String,
            val downloadedFormatted: String,
            val statusBengali: String
        ) : DownloadState

        data class Success(val modelFile: File, val preset: ModelPreset) : DownloadState

        data class Error(
            val message: String,
            val canRetry: Boolean = true
        ) : DownloadState
    }

    fun getModelsDir(context: Context): File {
        val dir = File(context.filesDir, "models")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        return dir
    }

    fun getModelFile(context: Context, preset: ModelPreset): File {
        return File(getModelsDir(context), preset.fileName)
    }

    fun getLegacyModelFile(context: Context): File {
        return File(context.filesDir, MODEL_FILE_NAME)
    }

    fun ensureModelMigration(context: Context) {
        val legacy = getLegacyModelFile(context)
        val defaultTarget = getModelFile(context, PRESETS[0])
        if (legacy.exists() && legacy.length() > 1_000_000L) {
            if (!defaultTarget.exists() || defaultTarget.length() == 0L) {
                legacy.copyTo(defaultTarget, overwrite = true)
            }
        }
    }

    fun isPresetDownloaded(context: Context, preset: ModelPreset): Boolean {
        ensureModelMigration(context)
        val file = getModelFile(context, preset)
        if (file.exists() && file.length() > 1_000_000L) return true
        if (preset.id == PRESETS[0].id) {
            val legacy = getLegacyModelFile(context)
            if (legacy.exists() && legacy.length() > 1_000_000L) return true
        }
        return false
    }

    fun getInstalledPresets(context: Context): List<ModelPreset> {
        return PRESETS.filter { isPresetDownloaded(context, it) }
    }

    fun deletePreset(context: Context, preset: ModelPreset): Boolean {
        val file = getModelFile(context, preset)
        var deleted = if (file.exists()) file.delete() else true
        if (preset.id == PRESETS[0].id) {
            val legacy = getLegacyModelFile(context)
            if (legacy.exists()) {
                deleted = legacy.delete() && deleted
            }
        }
        return deleted
    }

    fun cancelDownload(context: Context) {
        currentCall.getAndSet(null)?.cancel()
    }

    fun downloadModel(
        context: Context,
        preset: ModelPreset,
        customUrl: String? = null
    ): Flow<DownloadState> = flow {
        val targetFile = getModelFile(context, preset)
        val tempFile = File(getModelsDir(context), "${preset.fileName}.tmp")
        val url = if (!customUrl.isNullOrBlank()) customUrl.trim() else preset.url

        emit(
            DownloadState.Downloading(
                bytesRead = 0L,
                totalBytes = preset.sizeMb * 1024 * 1024,
                percentage = 0,
                speedBytesPerSec = 0L,
                speedFormatted = "0 KB/s",
                downloadedFormatted = "0.0 MB / ~${preset.sizeMb} MB",
                statusBengali = "সার্ভারের সাথে সংযোগ স্থাপন করা হচ্ছে..."
            )
        )

        try {
            val request = Request.Builder()
                .url(url)
                .header("User-Agent", "Mozilla/5.0 (Linux; Android 12; Mobile) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36")
                .header("Accept", "*/*")
                .header("Accept-Encoding", "identity")
                .build()

            val call = okHttpClient.newCall(request)
            currentCall.set(call)

            val response = call.execute()
            if (!response.isSuccessful) {
                val errorMsg = if (response.code == 429) {
                    "ডাউনলোড ব্যর্থ হয়েছে (HTTP 429 Too Many Requests)। Hugging Face রেট লিমিট ব্লক করেছে। অনুগ্রহ করে Qwen (ModelScope CDN) বেছে নিন অথবা নিচের বাটন দিয়ে লোকাল ফাইল ইমপোর্ট করুন।"
                } else {
                    "ডাউনলোড ব্যর্থ হয়েছে (HTTP ${response.code})। URL বা ইন্টারনেট সংযোগ পরীক্ষা করুন।"
                }
                emit(DownloadState.Error(message = errorMsg, canRetry = true))
                return@flow
            }

            val body = response.body
            if (body == null) {
                emit(DownloadState.Error("সার্ভার থেকে কোনো ফাইল ডেটা পাওয়া যায়নি।", canRetry = true))
                return@flow
            }

            val totalBytes = if (body.contentLength() > 0) {
                body.contentLength()
            } else {
                preset.sizeMb * 1024 * 1024
            }

            val startTime = System.currentTimeMillis()
            var lastSpeedCalcTime = startTime
            var bytesSinceLastSpeedCalc = 0L
            var currentSpeedBps = 0L

            body.byteStream().use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    var bytesRead = 0L
                    var read: Int
                    var lastEmittedPercentage = -1
                    var lastEmittedTime = 0L

                    while (input.read(buffer).also { read = it } != -1) {
                        coroutineContext.ensureActive()

                        output.write(buffer, 0, read)
                        bytesRead += read
                        bytesSinceLastSpeedCalc += read

                        val now = System.currentTimeMillis()
                        val elapsedSinceSpeedCalc = now - lastSpeedCalcTime
                        if (elapsedSinceSpeedCalc >= 500) {
                            currentSpeedBps = (bytesSinceLastSpeedCalc * 1000L) / elapsedSinceSpeedCalc
                            lastSpeedCalcTime = now
                            bytesSinceLastSpeedCalc = 0L
                        }

                        val percentage = if (totalBytes > 0) {
                            ((bytesRead * 100) / totalBytes).toInt().coerceIn(0, 100)
                        } else {
                            0
                        }

                        if (percentage != lastEmittedPercentage || (now - lastEmittedTime) >= 300) {
                            lastEmittedPercentage = percentage
                            lastEmittedTime = now

                            val mbDownloaded = bytesRead / (1024.0 * 1024.0)
                            val mbTotal = totalBytes / (1024.0 * 1024.0)
                            val speedFormatted = formatSpeed(currentSpeedBps)
                            val downloadedFormatted = "%.1f MB / %.1f MB".format(mbDownloaded, mbTotal)
                            val bengaliStatus = "ডাউনলোড হচ্ছে ($downloadedFormatted • $speedFormatted)"

                            emit(
                                DownloadState.Downloading(
                                    bytesRead = bytesRead,
                                    totalBytes = totalBytes,
                                    percentage = percentage,
                                    speedBytesPerSec = currentSpeedBps,
                                    speedFormatted = speedFormatted,
                                    downloadedFormatted = downloadedFormatted,
                                    statusBengali = bengaliStatus
                                )
                            )
                        }
                    }
                    output.flush()
                }
            }

            currentCall.set(null)

            if (tempFile.exists() && tempFile.length() > 0) {
                if (targetFile.exists()) {
                    targetFile.delete()
                }
                val renameSuccess = tempFile.renameTo(targetFile)
                if (renameSuccess && targetFile.exists()) {
                    targetFile.setReadable(true, false)
                    emit(DownloadState.Success(targetFile, preset))
                } else {
                    emit(DownloadState.Error("মডেল ফাইল স্টোরেজে সংরক্ষণ করতে ব্যর্থ হয়েছে।", canRetry = true))
                }
            } else {
                emit(DownloadState.Error("ডাউনলোডকৃত ফাইলটি খালি বা ক্ষতিগ্রস্ত।", canRetry = true))
            }
        } catch (e: Exception) {
            currentCall.set(null)
            if (tempFile.exists()) {
                tempFile.delete()
            }
            if (e is kotlinx.coroutines.CancellationException) {
                emit(DownloadState.Idle)
            } else {
                emit(
                    DownloadState.Error(
                        message = "ডাউনলোড ত্রুটি: ${e.localizedMessage ?: "নেটওয়ার্ক বিচ্ছিন্ন হয়েছে"}",
                        canRetry = true
                    )
                )
            }
        }
    }.flowOn(Dispatchers.IO)

    private fun formatSpeed(bytesPerSec: Long): String {
        return when {
            bytesPerSec >= 1024 * 1024 -> "%.1f MB/s".format(bytesPerSec / (1024.0 * 1024.0))
            bytesPerSec >= 1024 -> "%.0f KB/s".format(bytesPerSec / 1024.0)
            else -> "$bytesPerSec B/s"
        }
    }
}
