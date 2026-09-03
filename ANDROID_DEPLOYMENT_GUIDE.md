# 📱 Android App-এ Model Deploy করার সম্পূর্ণ গাইড

## 🎯 Overview

আপনার trained Qwen2.5-0.5B model Android app-এ ব্যবহার করার জন্য **3টি পদ্ধতি** আছে:

### পদ্ধতি ১: GGUF Format (সবচেয়ে সহজ) ⭐ RECOMMENDED
- Model size: ~300MB (quantized)
- Library: llama.cpp for Android
- Speed: Fast (CPU-তে চলে)
- Setup: সহজ

### পদ্ধতি ২: TensorFlow Lite (TFLite)
- Model size: ~200MB (quantized)
- Library: TensorFlow Lite
- Speed: Very fast (GPU support)
- Setup: মাঝারি জটিল

### পদ্ধতি ৩: ONNX Runtime
- Model size: ~400MB
- Library: ONNX Runtime Mobile
- Speed: Fast
- Setup: জটিল

**আমি পদ্ধতি ১ (GGUF) recommend করছি কারণ এটি সবচেয়ে সহজ এবং ভালো performance দেয়।**

---

# 🔥 পদ্ধতি ১: GGUF Format দিয়ে Android Deploy (RECOMMENDED)

## ধাপ ১: Model Convert করুন (GGUF Format-এ)

### 1.1 প্রয়োজনীয় Tools Install করুন

আপনার local machine-এ (Windows/Linux/Mac):

```bash
# Git clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Python requirements
pip install -r requirements.txt
```

### 1.2 আপনার Model Convert করুন

```bash
# ধাপ ১: Merge LoRA adapter with base model
python scripts/merge_lora.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --lora-adapter /path/to/qwen_nctb_final \
  --output-dir ./merged_model

# ধাপ ২: Convert to GGUF format
python convert_hf_to_gguf.py ./merged_model \
  --outtype f16 \
  --outfile qwen_nctb_bangla.gguf

# ধাপ ৩: Quantize করুন (size কমানোর জন্য)
./llama-quantize qwen_nctb_bangla.gguf qwen_nctb_bangla_q4_0.gguf q4_0
```

**Result**: `qwen_nctb_bangla_q4_0.gguf` (~300MB)

---

## ধাপ ২: Model GitHub-এ Upload করুন

### 2.1 GitHub Release তৈরি করুন

```bash
# আপনার repo-তে যান
cd ss_100m

# Model folder তৈরি করুন
mkdir android_models

# GGUF file copy করুন
cp /path/to/qwen_nctb_bangla_q4_0.gguf android_models/

# Git add করুন
git add android_models/qwen_nctb_bangla_q4_0.gguf
git commit -m "Add GGUF model for Android deployment"
git push origin main
```

### 2.2 GitHub Release তৈরি করুন (Large File-এর জন্য)

যেহেতু model file বড় (300MB+), GitHub Release ব্যবহার করুন:

1. GitHub repo-তে যান: https://github.com/kajshikhi49-afk/ss_100m
2. **Releases** → **Create a new release**
3. Tag: `v1.0.0-android`
4. Title: "Bengali NCTB Model (Android GGUF)"
5. Upload: `qwen_nctb_bangla_q4_0.gguf`
6. Publish release

**Download Link পাবেন:**
```
https://github.com/kajshikhi49-afk/ss_100m/releases/download/v1.0.0-android/qwen_nctb_bangla_q4_0.gguf
```

---

## ধাপ ৩: Android App তৈরি করুন (Kotlin)

### 3.1 Project Setup

**build.gradle.kts (Module level):**

```kotlin
dependencies {
    // llama.cpp Android library
    implementation("com.github.moxinan:llama.cpp:v1.0.0")
    
    // অথবা জনপ্রিয় wrapper:
    implementation("com.github.moxinan:llamacpp-android:0.1.0")
    
    // Coroutines (async download এর জন্য)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // OkHttp (file download)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    
    // WorkManager (background download)
    implementation("androidx.work:work-runtime-ktx:2.9.0")
}
```

**AndroidManifest.xml:**

```xml
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
```

---

### 3.2 Model Manager Class তৈরি করুন

**ModelManager.kt:**

```kotlin
package com.example.nctbai

import android.content.Context
import kotlinx.coroutines.*
import okhttp3.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ModelManager(private val context: Context) {
    
    private val modelUrl = "https://github.com/kajshikhi49-afk/ss_100m/releases/download/v1.0.0-android/qwen_nctb_bangla_q4_0.gguf"
    private val modelFileName = "qwen_nctb_bangla_q4_0.gguf"
    private val modelFile: File
        get() = File(context.filesDir, modelFileName)
    
    // Check if model আগে থেকে download করা আছে
    fun isModelDownloaded(): Boolean {
        return modelFile.exists() && modelFile.length() > 0
    }
    
    // Model download করুন (progress callback সহ)
    suspend fun downloadModel(
        onProgress: (percent: Int) -> Unit,
        onComplete: () -> Unit,
        onError: (error: String) -> Unit
    ) = withContext(Dispatchers.IO) {
        
        if (isModelDownloaded()) {
            withContext(Dispatchers.Main) {
                onComplete()
            }
            return@withContext
        }
        
        val client = OkHttpClient()
        val request = Request.Builder().url(modelUrl).build()
        
        try {
            val response = client.newCall(request).execute()
            
            if (!response.isSuccessful) {
                throw IOException("Download failed: ${response.code}")
            }
            
            val body = response.body ?: throw IOException("Response body is null")
            val contentLength = body.contentLength()
            val inputStream = body.byteStream()
            val outputStream = FileOutputStream(modelFile)
            
            val buffer = ByteArray(8192)
            var downloaded: Long = 0
            var lastProgress = 0
            
            while (true) {
                val read = inputStream.read(buffer)
                if (read == -1) break
                
                outputStream.write(buffer, 0, read)
                downloaded += read
                
                // Progress calculate করুন
                val progress = ((downloaded * 100) / contentLength).toInt()
                if (progress != lastProgress) {
                    withContext(Dispatchers.Main) {
                        onProgress(progress)
                    }
                    lastProgress = progress
                }
            }
            
            outputStream.close()
            inputStream.close()
            
            withContext(Dispatchers.Main) {
                onComplete()
            }
            
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                onError(e.message ?: "Unknown error")
            }
        }
    }
    
    // Model file path return করুন
    fun getModelPath(): String {
        return modelFile.absolutePath
    }
    
    // Model delete করুন (if needed)
    fun deleteModel() {
        if (modelFile.exists()) {
            modelFile.delete()
        }
    }
    
    // Model size check করুন
    fun getModelSize(): Long {
        return if (modelFile.exists()) modelFile.length() else 0
    }
}
```

---

### 3.3 AI Inference Class তৈরি করুন

**AIInference.kt:**

```kotlin
package com.example.nctbai

import android.content.Context
import com.github.moxinan.llamacpp.LlamaAndroid

class AIInference(private val context: Context) {
    
    private var llama: LlamaAndroid? = null
    private var isModelLoaded = false
    
    // Model load করুন (প্রথমবার app খুললে)
    suspend fun loadModel(modelPath: String): Boolean {
        return try {
            llama = LlamaAndroid.load(modelPath).apply {
                // Model configuration
                setNThreads(4)  // CPU threads
                setTemperature(0.7f)
                setTopK(40)
                setTopP(0.9f)
                setRepeatPenalty(1.1f)
            }
            isModelLoaded = true
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    // প্রশ্নের উত্তর generate করুন
    suspend fun generateAnswer(question: String): String {
        if (!isModelLoaded || llama == null) {
            return "Model not loaded!"
        }
        
        // Prompt format (Qwen instruction format)
        val prompt = """<|im_start|>system
You are a helpful AI assistant for Bengali education. Answer questions based on NCTB curriculum.<|im_end|>
<|im_start|>user
$question<|im_end|>
<|im_start|>assistant
"""
        
        return try {
            val response = llama?.generate(prompt, maxTokens = 256) ?: "Error generating response"
            
            // Extract only assistant's answer
            response.substringAfter("<|im_start|>assistant\n")
                   .substringBefore("<|im_end|>")
                   .trim()
            
        } catch (e: Exception) {
            "Error: ${e.message}"
        }
    }
    
    // Model unload করুন (memory free করার জন্য)
    fun unloadModel() {
        llama?.close()
        llama = null
        isModelLoaded = false
    }
}
```

---

### 3.4 Main Activity (UI)

**MainActivity.kt:**

```kotlin
package com.example.nctbai

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    
    private lateinit var modelManager: ModelManager
    private lateinit var aiInference: AIInference
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        modelManager = ModelManager(this)
        aiInference = AIInference(this)
        
        setContent {
            NCTBAIApp()
        }
    }
    
    @Composable
    fun NCTBAIApp() {
        var isDownloading by remember { mutableStateOf(false) }
        var downloadProgress by remember { mutableStateOf(0) }
        var isModelReady by remember { mutableStateOf(modelManager.isModelDownloaded()) }
        var question by remember { mutableStateOf("") }
        var answer by remember { mutableStateOf("") }
        var isGenerating by remember { mutableStateOf(false) }
        
        val scope = rememberCoroutineScope()
        
        // Model load করুন (if already downloaded)
        LaunchedEffect(isModelReady) {
            if (isModelReady && !isDownloading) {
                aiInference.loadModel(modelManager.getModelPath())
            }
        }
        
        Scaffold(
            topBar = {
                TopAppBar(title = { Text("NCTB AI Assistant") })
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                
                // Download Section
                if (!isModelReady) {
                    Text("AI Model Download করুন", style = MaterialTheme.typography.headlineSmall)
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    if (isDownloading) {
                        LinearProgressIndicator(
                            progress = downloadProgress / 100f,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Text("Downloading: $downloadProgress%")
                    } else {
                        Button(onClick = {
                            scope.launch {
                                isDownloading = true
                                modelManager.downloadModel(
                                    onProgress = { progress ->
                                        downloadProgress = progress
                                    },
                                    onComplete = {
                                        isDownloading = false
                                        isModelReady = true
                                    },
                                    onError = { error ->
                                        isDownloading = false
                                        // Show error dialog
                                    }
                                )
                            }
                        }) {
                            Text("Model Download করুন (~300 MB)")
                        }
                    }
                }
                
                // Chat Section
                if (isModelReady && !isDownloading) {
                    Text("প্রশ্ন করুন:", style = MaterialTheme.typography.headlineSmall)
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    OutlinedTextField(
                        value = question,
                        onValueChange = { question = it },
                        label = { Text("আপনার প্রশ্ন লিখুন") },
                        modifier = Modifier.fillMaxWidth(),
                        maxLines = 3
                    )
                    
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    Button(
                        onClick = {
                            scope.launch {
                                isGenerating = true
                                answer = aiInference.generateAnswer(question)
                                isGenerating = false
                            }
                        },
                        enabled = !isGenerating && question.isNotBlank()
                    ) {
                        Text(if (isGenerating) "উত্তর তৈরি হচ্ছে..." else "উত্তর পান")
                    }
                    
                    if (isGenerating) {
                        CircularProgressIndicator()
                    }
                    
                    if (answer.isNotBlank()) {
                        Spacer(modifier = Modifier.height(24.dp))
                        Card(
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Text("উত্তর:", style = MaterialTheme.typography.titleMedium)
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(answer)
                            }
                        }
                    }
                }
            }
        }
    }
}
```

---

## ধাপ ৪: User Flow

### 📱 User Experience:

1. **প্রথমবার App খুললে:**
   ```
   ┌─────────────────────────┐
   │  NCTB AI Assistant      │
   ├─────────────────────────┤
   │                         │
   │  AI Model Download করুন │
   │                         │
   │  [Download Button]      │
   │  (~300 MB)              │
   │                         │
   └─────────────────────────┘
   ```

2. **Download চলাকালীন:**
   ```
   ┌─────────────────────────┐
   │  NCTB AI Assistant      │
   ├─────────────────────────┤
   │                         │
   │  Downloading: 47%       │
   │  ████████░░░░░░░░       │
   │                         │
   │  Please wait...         │
   │                         │
   └─────────────────────────┘
   ```

3. **Download সম্পন্ন হলে:**
   ```
   ┌─────────────────────────┐
   │  NCTB AI Assistant      │
   ├─────────────────────────┤
   │                         │
   │  প্রশ্ন করুন:           │
   │  ┌───────────────────┐  │
   │  │ আপনার প্রশ্ন...   │  │
   │  └───────────────────┘  │
   │                         │
   │  [উত্তর পান]           │
   │                         │
   │  ┌───────────────────┐  │
   │  │ উত্তর: ...        │  │
   │  └───────────────────┘  │
   └─────────────────────────┘
   ```

---

## ধাপ ৫: Optimization & Best Practices

### 5.1 Model Caching

```kotlin
// SharedPreferences দিয়ে track করুন
class ModelPreferences(context: Context) {
    private val prefs = context.getSharedPreferences("model_prefs", Context.MODE_PRIVATE)
    
    fun isModelDownloaded(): Boolean {
        return prefs.getBoolean("model_downloaded", false)
    }
    
    fun setModelDownloaded(downloaded: Boolean) {
        prefs.edit().putBoolean("model_downloaded", downloaded).apply()
    }
    
    fun getModelVersion(): String {
        return prefs.getString("model_version", "") ?: ""
    }
    
    fun setModelVersion(version: String) {
        prefs.edit().putString("model_version", version).apply()
    }
}
```

### 5.2 Background Download (WorkManager)

```kotlin
class ModelDownloadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {
    
    override suspend fun doWork(): Result {
        val modelManager = ModelManager(applicationContext)
        
        return try {
            modelManager.downloadModel(
                onProgress = { progress ->
                    setProgress(workDataOf("progress" to progress))
                },
                onComplete = {
                    // Notify user
                },
                onError = { error ->
                    // Handle error
                }
            )
            Result.success()
        } catch (e: Exception) {
            Result.failure()
        }
    }
}
```

### 5.3 Model Update Check

```kotlin
suspend fun checkForModelUpdate(): Boolean {
    // GitHub API দিয়ে latest release check করুন
    val latestVersion = fetchLatestVersionFromGitHub()
    val currentVersion = modelPreferences.getModelVersion()
    
    return latestVersion != currentVersion
}
```

---

## 📊 Expected Performance

### Model Specs:
- **Size**: ~300MB (q4_0 quantized)
- **RAM Usage**: ~500MB (runtime)
- **Inference Speed**: 
  - Mid-range phone: ~20-30 tokens/sec
  - High-end phone: ~40-60 tokens/sec
- **First Load Time**: 2-5 seconds
- **Answer Generation**: 3-10 seconds

### Minimum Requirements:
- **Android**: 8.0+ (API 26+)
- **RAM**: 3GB+
- **Storage**: 500MB free space
- **CPU**: Quad-core 1.5GHz+

---

## 🎯 Alternative: Smaller Model Size

যদি 300MB বড় মনে হয়:

### Option A: More Quantization
```bash
# Q3_K_S (smaller, slightly less quality)
./llama-quantize qwen_nctb_bangla.gguf qwen_nctb_bangla_q3_k_s.gguf q3_k_s
# Size: ~200MB
```

### Option B: Dynamic Download
```kotlin
// শুধু যেই subject-এর প্রশ্ন করবে সেটার model download করুন
val subjects = listOf("bangla", "math", "science", "english")
// প্রতিটি ~80MB
```

---

## 📦 Complete Project Structure

```
NCTBAIApp/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/example/nctbai/
│   │   │   │   ├── MainActivity.kt
│   │   │   │   ├── ModelManager.kt
│   │   │   │   ├── AIInference.kt
│   │   │   │   ├── ModelPreferences.kt
│   │   │   │   └── ModelDownloadWorker.kt
│   │   │   ├── res/
│   │   │   └── AndroidManifest.xml
│   │   └── build.gradle.kts
│   └── build.gradle.kts
└── README.md
```

---

## 🚀 Publishing Checklist

- [ ] Model GGUF format-এ convert করা হয়েছে
- [ ] GitHub Release-এ upload করা হয়েছে
- [ ] Android app তৈরি করা হয়েছে
- [ ] Download functionality test করা হয়েছে
- [ ] Inference test করা হয়েছে
- [ ] Multiple devices-এ test করা হয়েছে
- [ ] Play Store guidelines follow করা হয়েছে

---

## 💡 Pro Tips

1. **WiFi-Only Download**: User-কে option দিন WiFi-তে download করার
2. **Resume Support**: Download interrupted হলে resume করার feature
3. **Model Versioning**: নতুন version আসলে update করার system
4. **Offline Mode**: একবার download হলে সব offline কাজ করবে
5. **Usage Analytics**: কোন প্রশ্ন বেশি আসে track করুন

---

**এই complete guide follow করে আপনি একটি production-ready Android AI app বানাতে পারবেন!** 🚀

**Repository**: https://github.com/kajshikhi49-afk/ss_100m
