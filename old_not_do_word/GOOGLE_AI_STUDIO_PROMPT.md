# 🤖 Google AI Studio Prompt - NCTB AI Assistant Android App

## 📋 Complete Prompt for AI Studio

Copy and paste this entire prompt into Google AI Studio (or any AI code generator):

---

## 🎯 PROMPT START

```
Create a complete Android app in Kotlin with Jetpack Compose for an offline AI assistant that helps students with NCTB (Bangladesh) curriculum questions.

### APP REQUIREMENTS:

#### 1. APP OVERVIEW:
- **Name**: NCTB AI Assistant
- **Package**: com.nctb.ai.assistant
- **Min SDK**: 26 (Android 8.0)
- **Target SDK**: 34 (Android 14)
- **Language**: Kotlin
- **UI**: Jetpack Compose with Material 3
- **Architecture**: MVVM with Repository pattern

#### 2. KEY FEATURES:
1. **First Launch**: Show model download screen
2. **Model Download**: Download GGUF model from GitHub Release with progress bar
3. **Offline AI Chat**: Ask questions and get answers (offline after download)
4. **Chat History**: Save previous conversations locally
5. **Bengali Language Support**: Full Bengali UI and content support
6. **Model Management**: Check for updates, delete model, view model info

#### 3. DEPENDENCIES (build.gradle.kts - Module level):

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.nctb.ai.assistant"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.nctb.ai.assistant"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    
    kotlinOptions {
        jvmTarget = "17"
    }
    
    buildFeatures {
        compose = true
    }
}

dependencies {
    // AndroidX Core
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    implementation("androidx.activity:activity-compose:1.8.2")

    // Jetpack Compose
    implementation(platform("androidx.compose:compose-bom:2024.02.00"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    
    // Navigation
    implementation("androidx.navigation:navigation-compose:2.7.6")
    
    // ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.7.0")
    
    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    
    // OkHttp (file download)
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    
    // Room Database (chat history)
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    annotationProcessor("androidx.room:room-compiler:2.6.1")
    
    // DataStore (preferences)
    implementation("androidx.datastore:datastore-preferences:1.0.0")
    
    // llama.cpp for Android (GGUF support)
    implementation("de.kherud:llama:2.2.1")  // Or latest version
    
    // WorkManager (background tasks)
    implementation("androidx.work:work-runtime-ktx:2.9.0")
}
```

#### 4. ANDROID MANIFEST (AndroidManifest.xml):

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools">

    <uses-permission android:name="android.permission.INTERNET"/>
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE"/>
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"
        android:maxSdkVersion="28" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"
        android:maxSdkVersion="32" />
    
    <application
        android:name=".NCTBApplication"
        android:allowBackup="true"
        android:dataExtractionRules="@xml/data_extraction_rules"
        android:fullBackupContent="@xml/backup_rules"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/Theme.NCTBAIAssistant"
        android:usesCleartextTraffic="true"
        tools:targetApi="31">
        
        <activity
            android:name=".MainActivity"
            android:exported="true"
            android:theme="@style/Theme.NCTBAIAssistant"
            android:windowSoftInputMode="adjustResize">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

#### 5. KEY CONFIGURATION VALUES:

**IMPORTANT - MODEL DOWNLOAD URL:**
```kotlin
// This will be replaced with actual GitHub Release link
const val MODEL_DOWNLOAD_URL = "REPLACE_WITH_GITHUB_RELEASE_URL"
const val MODEL_FILE_NAME = "qwen_nctb_bangla_q4_0.gguf"
const val MODEL_VERSION = "1.0.0"
```

**Create a file: `app/src/main/java/com/nctb/ai/assistant/util/Constants.kt`**
```kotlin
package com.nctb.ai.assistant.util

object Constants {
    // REPLACE THIS URL with your GitHub Release URL
    const val MODEL_DOWNLOAD_URL = "REPLACE_WITH_GITHUB_RELEASE_URL"
    
    const val MODEL_FILE_NAME = "qwen_nctb_bangla_q4_0.gguf"
    const val MODEL_VERSION = "1.0.0"
    
    // AI Configuration
    const val MAX_TOKENS = 256
    const val TEMPERATURE = 0.7f
    const val TOP_K = 40
    const val TOP_P = 0.9f
    const val REPEAT_PENALTY = 1.1f
    const val N_THREADS = 4
    
    // Prompt Template
    const val SYSTEM_PROMPT = "You are a helpful AI assistant for Bengali education. Answer questions based on NCTB curriculum."
}
```

#### 6. ARCHITECTURE COMPONENTS:

Create these files with complete implementation:

**A. Data Layer:**
- `data/local/AppDatabase.kt` - Room database for chat history
- `data/local/ChatDao.kt` - DAO for chat operations
- `data/model/ChatMessage.kt` - Data class for messages
- `data/repository/ModelRepository.kt` - Handle model download/management
- `data/repository/ChatRepository.kt` - Handle chat operations

**B. Domain Layer:**
- `domain/model/DownloadState.kt` - Sealed class for download states
- `domain/usecase/DownloadModelUseCase.kt` - Download logic
- `domain/usecase/GenerateAnswerUseCase.kt` - AI inference logic

**C. Presentation Layer:**
- `ui/screen/SplashScreen.kt` - App splash screen
- `ui/screen/DownloadScreen.kt` - Model download UI
- `ui/screen/ChatScreen.kt` - Main chat interface
- `ui/screen/SettingsScreen.kt` - Settings and model management
- `ui/viewmodel/DownloadViewModel.kt` - Download screen logic
- `ui/viewmodel/ChatViewModel.kt` - Chat screen logic
- `ui/navigation/NavGraph.kt` - Navigation setup

**D. Utility:**
- `util/Constants.kt` - App constants
- `util/ModelManager.kt` - GGUF model management
- `util/AIInference.kt` - llama.cpp wrapper

#### 7. KEY UI SCREENS REQUIREMENTS:

**A. Splash Screen:**
- Show app logo and name
- Check if model is downloaded
- Navigate to Download or Chat screen accordingly

**B. Download Screen:**
```
┌────────────────────────────────────┐
│  NCTB AI Assistant                 │
├────────────────────────────────────┤
│                                    │
│  📚 বাংলাদেশ শিক্ষা AI সহায়ক      │
│                                    │
│  এই অ্যাপ ব্যবহার করতে AI মডেল    │
│  ডাউনলোড করতে হবে।                │
│                                    │
│  মডেল সাইজ: ~300 MB               │
│  WiFi সংযোগ প্রয়োজন                │
│                                    │
│  [Download Now Button]             │
│                                    │
│  Progress: ████████░░░░ 67%       │
│  Downloaded: 201 MB / 300 MB      │
│                                    │
└────────────────────────────────────┘
```

**C. Chat Screen:**
```
┌────────────────────────────────────┐
│  ← NCTB AI Assistant        ⋮     │
├────────────────────────────────────┤
│                                    │
│  👤 User:                          │
│  বাংলা বর্ণমালায় মোট কয়টি বর্ণ?  │
│                                    │
│  🤖 AI:                            │
│  বাংলা বর্ণমালায় মোট ৫০টি বর্ণ    │
│  আছে। এর মধ্যে স্বরবর্ণ ১১টি এবং   │
│  ব্যঞ্জনবর্ণ ৩৯টি।                 │
│                                    │
│  👤 User:                          │
│  পদ্মা নদী কোথায়?                │
│                                    │
│  🤖 AI: [Typing...]                │
│                                    │
├────────────────────────────────────┤
│  ✍️  আপনার প্রশ্ন লিখুন...    [→] │
└────────────────────────────────────┘
```

**D. Settings Screen:**
- Model Info (size, version, location)
- Delete Model button
- Check for Updates button
- About App
- Privacy Policy

#### 8. IMPORTANT IMPLEMENTATION DETAILS:

**A. Model Download with Resume Support:**
```kotlin
suspend fun downloadModel(
    url: String,
    file: File,
    onProgress: (downloaded: Long, total: Long) -> Unit
): Result<Unit> {
    return withContext(Dispatchers.IO) {
        try {
            val request = Request.Builder()
                .url(url)
                .apply {
                    if (file.exists() && file.length() > 0) {
                        addHeader("Range", "bytes=${file.length()}-")
                    }
                }
                .build()
            
            // Implementation with resume support
            // ... (full implementation needed)
            
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
```

**B. AI Inference with llama.cpp:**
```kotlin
class AIInference(private val context: Context) {
    private var model: LlamaModel? = null
    
    fun loadModel(modelPath: String): Boolean {
        return try {
            model = LlamaModel(modelPath).apply {
                // Configure model parameters
            }
            true
        } catch (e: Exception) {
            false
        }
    }
    
    suspend fun generate(prompt: String): String {
        return withContext(Dispatchers.Default) {
            val fullPrompt = buildPrompt(prompt)
            model?.generate(fullPrompt) ?: "Model not loaded"
        }
    }
    
    private fun buildPrompt(userQuestion: String): String {
        return """<|im_start|>system
${Constants.SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
$userQuestion<|im_end|>
<|im_start|>assistant
"""
    }
}
```

**C. Chat History with Room:**
```kotlin
@Entity(tableName = "chat_messages")
data class ChatMessage(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val message: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis()
)

@Dao
interface ChatDao {
    @Query("SELECT * FROM chat_messages ORDER BY timestamp ASC")
    fun getAllMessages(): Flow<List<ChatMessage>>
    
    @Insert
    suspend fun insertMessage(message: ChatMessage)
    
    @Query("DELETE FROM chat_messages")
    suspend fun clearAll()
}
```

#### 9. UI THEME (Material 3):

**Colors:**
- Primary: #1976D2 (Blue)
- Secondary: #4CAF50 (Green)
- Background: #FFFFFF (White)
- Surface: #F5F5F5 (Light Gray)
- Error: #D32F2F (Red)

**Bengali Font Support:**
- Use Noto Sans Bengali or system default
- Ensure proper Unicode support

#### 10. ERROR HANDLING:

Handle these scenarios:
- No internet connection (during download)
- Insufficient storage space
- Model file corrupted
- Model loading failed
- Generation timeout
- App crash recovery

#### 11. PERFORMANCE OPTIMIZATION:

- Use lazy loading for chat history
- Implement proper cancellation for coroutines
- Memory-efficient image loading
- Background processing for AI inference
- Proper lifecycle management

#### 12. TESTING REQUIREMENTS:

Create basic tests for:
- Model download functionality
- AI inference (mock)
- Chat repository
- ViewModel logic

#### 13. APP STRINGS (strings.xml):

Include Bengali strings:
```xml
<resources>
    <string name="app_name">NCTB AI Assistant</string>
    <string name="download_model">মডেল ডাউনলোড করুন</string>
    <string name="downloading">ডাউনলোড হচ্ছে...</string>
    <string name="download_complete">ডাউনলোড সম্পন্ন!</string>
    <string name="ask_question">প্রশ্ন করুন</string>
    <string name="type_message">আপনার প্রশ্ন লিখুন...</string>
    <string name="model_info">মডেল তথ্য</string>
    <string name="delete_model">মডেল মুছুন</string>
    <string name="check_updates">আপডেট চেক করুন</string>
</resources>
```

#### 14. PROGUARD RULES (proguard-rules.pro):

```proguard
# llama.cpp
-keep class de.kherud.llama.** { *; }

# Room
-keep class * extends androidx.room.RoomDatabase
-keep @androidx.room.Entity class *
-keepclassmembers class * extends androidx.room.RoomDatabase {
    public static ** getInstance(...);
}

# OkHttp
-dontwarn okhttp3.**
-dontwarn okio.**
```

#### 15. ADDITIONAL FEATURES (OPTIONAL):

- Share answer feature
- Copy answer to clipboard
- Voice input support
- Dark mode support
- Multiple language support (Bangla/English toggle)
- Export chat history
- Bookmark important Q&A

#### 16. RELEASE CONFIGURATION:

**App Signing:**
- Generate keystore for release
- Configure signing in build.gradle

**Version Management:**
```kotlin
defaultConfig {
    versionCode = 1  // Increment for each release
    versionName = "1.0.0"  // User-visible version
}
```

#### 17. PERMISSIONS EXPLANATION (in app):

Show permission dialog explaining:
- Internet: For model download only
- Storage: To save model file locally
- No data collection
- 100% offline after download

#### 18. README FOR GITHUB:

Include:
- App screenshots
- Features list
- Installation guide
- Build instructions
- License (MIT/Apache 2.0)

---

### FINAL OUTPUT REQUIREMENTS:

Generate a complete, production-ready Android app with:
1. ✅ All source files organized properly
2. ✅ Complete build.gradle files
3. ✅ Working UI with Jetpack Compose
4. ✅ Model download functionality
5. ✅ AI inference integration
6. ✅ Chat history persistence
7. ✅ Error handling
8. ✅ Bengali language support
9. ✅ Material 3 design
10. ✅ Ready to compile and run

**IMPORTANT NOTE:**
After generation, I only need to replace `REPLACE_WITH_GITHUB_RELEASE_URL` in Constants.kt with my actual model download link.

Make the app professional, user-friendly, and optimized for performance. Include proper comments in the code.
```

## 🎯 PROMPT END

---

## 📝 Instructions for You:

1. **Copy the entire prompt above** (from "PROMPT START" to "PROMPT END")

2. **Paste into Google AI Studio** or your preferred AI code generator

3. **Wait for generation** (may take 2-5 minutes)

4. **Download the generated project**

5. **Replace the URL** in `Constants.kt`:
   ```kotlin
   const val MODEL_DOWNLOAD_URL = "YOUR_GITHUB_RELEASE_URL_HERE"
   ```

6. **Build and test!**

---

## 🔗 After Model is Ready:

When you convert your model to GGUF and upload to GitHub Release, you'll get a URL like:

```
https://github.com/kajshikhi49-afk/ss_100m/releases/download/v1.0.0-android/qwen_nctb_bangla_q4_0.gguf
```

Just paste this URL in Constants.kt and rebuild!

---

## ✅ What This Prompt Will Generate:

- ✅ Complete Android Studio project
- ✅ All Kotlin source files
- ✅ Gradle configuration
- ✅ UI with Jetpack Compose
- ✅ Model download feature
- ✅ Chat interface
- ✅ Settings screen
- ✅ Database setup
- ✅ Ready to compile!

**You'll only need to add your model URL and you're done!** 🚀

---

**File saved as**: `GOOGLE_AI_STUDIO_PROMPT.md`
