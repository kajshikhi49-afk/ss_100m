package com.example

import android.app.ActivityManager
import android.content.Context
import android.os.Build

/**
 * Hardware and Architecture Diagnostics utility for Android on-device LLM inference.
 * Provides ABI checks (32-bit armeabi-v7a vs 64-bit arm64-v8a) and memory profiling.
 */
object HardwareInfo {

    val supportedAbis: List<String>
        get() = Build.SUPPORTED_ABIS.toList()

    val primaryAbi: String
        get() = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"

    val is64Bit: Boolean
        get() = Build.SUPPORTED_ABIS.any { it.contains("64") }

    val deviceModel: String
        get() = "${Build.MANUFACTURER} ${Build.MODEL}"

    val androidVersion: String
        get() = "Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})"

    /**
     * Returns a pair of (availableMemBytes, totalMemBytes).
     */
    fun getMemoryInfo(context: Context): Pair<Long, Long> {
        return try {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            if (am != null) {
                val memInfo = ActivityManager.MemoryInfo()
                am.getMemoryInfo(memInfo)
                Pair(memInfo.availMem, memInfo.totalMem)
            } else {
                Pair(0L, 0L)
            }
        } catch (e: Exception) {
            Pair(0L, 0L)
        }
    }

    /**
     * Formats bytes to human-readable MB / GB.
     */
    fun formatBytes(bytes: Long): String {
        return when {
            bytes >= 1024 * 1024 * 1024 -> "%.1f GB".format(bytes / (1024.0 * 1024.0 * 1024.0))
            bytes >= 1024 * 1024 -> "%.0f MB".format(bytes / (1024.0 * 1024.0))
            else -> "$bytes B"
        }
    }

    /**
     * Explanation and CMake compile instructions for 32-bit armeabi-v7a (e.g. Unisoc SC9863A / itel A60).
     */
    const val ARMV7_COMPILE_COMMAND = """# To build 32-bit libllama.so for armeabi-v7a (Unisoc SC9863A):
cmake -B build-armv7 \
  -DCMAKE_TOOLCHAIN_FILE=${'$'}ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=armeabi-v7a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF

cmake --build build-armv7 --target llama --config Release

# Copy compiled shared library into project:
mkdir -p app/src/main/jniLibs/armeabi-v7a
cp build-armv7/bin/libllama.so app/src/main/jniLibs/armeabi-v7a/libllama.so"""
}
