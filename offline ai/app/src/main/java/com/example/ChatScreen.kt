package com.example

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.navigationBars
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBars
import androidx.compose.foundation.layout.union
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Cancel
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeveloperMode
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.SmartToy
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.ui.theme.AiBubbleDark
import com.example.ui.theme.AiBubbleLight
import com.example.ui.theme.ChatBackgroundDark
import com.example.ui.theme.ChatBackgroundLight
import com.example.ui.theme.EmeraldPrimary
import com.example.ui.theme.OnlineGreen
import com.example.ui.theme.UserBubbleDark
import com.example.ui.theme.UserBubbleLight
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Modern Jetpack Compose Material 3 Chat UI with real-time token streaming,
 * multi-model management, ABI diagnostics, and robust error dialogs.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    messages: List<ChatMessage>,
    isGenerating: Boolean,
    errorMessage: String?,
    showDiagnosticsDialog: Boolean,
    showModelInfoDialog: Boolean,
    showModelSwitcherDialog: Boolean,
    activePreset: ModelDownloader.ModelPreset,
    installedPresets: List<ModelDownloader.ModelPreset>,
    onSendMessage: (String) -> Unit,
    onClearChat: () -> Unit,
    onDismissError: () -> Unit,
    onToggleDiagnostics: (Boolean) -> Unit,
    onToggleModelInfo: (Boolean) -> Unit,
    onToggleModelSwitcher: (Boolean) -> Unit,
    onSwitchModel: (ModelDownloader.ModelPreset) -> Unit,
    onDownloadPreset: (ModelDownloader.ModelPreset) -> Unit,
    onDeletePreset: (ModelDownloader.ModelPreset) -> Unit,
    onImportModel: (Uri) -> Unit,
    modifier: Modifier = Modifier
) {
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    val isDark = isSystemInDarkTheme()

    // Real-time auto-scroll to the bottom as tokens stream in
    LaunchedEffect(messages.size, messages.lastOrNull()?.text, isGenerating) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    val samplePrompts = listOf(
        "কেমন আছো?",
        "বাংলাদেশের রাজধানী কী?",
        "কম্পিউটার কীভাবে কাজ করে?",
        "বাংলা সাহিত্যের ইতিহাস সম্পর্কে বলো"
    )

    Scaffold(
        modifier = modifier.fillMaxSize(),
        contentWindowInsets = WindowInsets.statusBars,
        topBar = {
            TopAppBar(
                title = {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        // AI Avatar
                        Box(
                            modifier = Modifier
                                .size(42.dp)
                                .clip(CircleShape)
                                .background(Color.White.copy(alpha = 0.2f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Text(
                                text = activePreset.shortName.take(1),
                                color = Color.White,
                                fontWeight = FontWeight.Bold,
                                fontSize = 20.sp
                            )
                        }

                        Column(
                            modifier = Modifier.clickable { onToggleModelSwitcher(true) }
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                Text(
                                    text = activePreset.shortName,
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold,
                                    color = Color.White,
                                    maxLines = 1,
                                    overflow = TextOverflow.Ellipsis
                                )
                                Text(
                                    text = "▼",
                                    fontSize = 10.sp,
                                    color = Color.White.copy(alpha = 0.8f)
                                )
                            }

                            // Green status badge: "১০০% অফলাইন • অন-ডিভাইস"
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(5.dp)
                            ) {
                                Box(
                                    modifier = Modifier
                                        .size(8.dp)
                                        .clip(CircleShape)
                                        .background(OnlineGreen)
                                )
                                Text(
                                    text = "১০০% অফলাইন • অন-ডিভাইস",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color.White.copy(alpha = 0.95f),
                                    fontWeight = FontWeight.Medium
                                )
                            }
                        }
                    }
                },
                actions = {
                    IconButton(
                        onClick = { onToggleModelSwitcher(true) },
                        modifier = Modifier.testTag("switch_model_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "মডেল পরিবর্তন",
                            tint = Color.White
                        )
                    }
                    IconButton(
                        onClick = { onToggleDiagnostics(true) },
                        modifier = Modifier.testTag("diagnostics_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.DeveloperMode,
                            contentDescription = "হার্ডওয়্যার ডায়াগনস্টিকস",
                            tint = Color.White
                        )
                    }
                    IconButton(
                        onClick = { onToggleModelInfo(true) },
                        modifier = Modifier.testTag("model_info_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Info,
                            contentDescription = "মডেলের তথ্য",
                            tint = Color.White
                        )
                    }
                    if (messages.isNotEmpty()) {
                        IconButton(
                            onClick = onClearChat,
                            modifier = Modifier.testTag("clear_chat_button")
                        ) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "চ্যাট মুছুন",
                                tint = Color.White
                            )
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = EmeraldPrimary
                )
            )
        },
        bottomBar = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .windowInsetsPadding(WindowInsets.navigationBars.union(WindowInsets.ime))
            ) {
                // Quick prompt suggestions when input is empty and not generating
                AnimatedVisibility(visible = messages.isNotEmpty() && !isGenerating && inputText.isEmpty()) {
                    LazyRow(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(if (isDark) Color(0xFF1F2C34) else Color(0xFFF0F2F5))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        items(samplePrompts) { prompt ->
                            Surface(
                                shape = RoundedCornerShape(16.dp),
                                color = MaterialTheme.colorScheme.surface,
                                shadowElevation = 1.dp,
                                modifier = Modifier.clickable { inputText = prompt }
                            ) {
                                Text(
                                    text = prompt,
                                    style = MaterialTheme.typography.bodySmall,
                                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                                    color = MaterialTheme.colorScheme.onSurface
                                )
                            }
                        }
                    }
                }

                // Bottom Input Bar
                Surface(
                    color = if (isDark) Color(0xFF1F2C34) else Color(0xFFF0F2F5),
                    shadowElevation = 4.dp,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 8.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        OutlinedTextField(
                            value = inputText,
                            onValueChange = { inputText = it },
                            placeholder = {
                                Text(
                                    text = if (isGenerating) "মডেল উত্তর লিখছে..." else "প্রশ্ন লিখুন... (Type in Bengali)",
                                    fontSize = 14.sp
                                )
                            },
                            enabled = !isGenerating,
                            modifier = Modifier
                                .weight(1f)
                                .testTag("chat_input"),
                            shape = RoundedCornerShape(24.dp),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedContainerColor = if (isDark) Color(0xFF2A3942) else Color.White,
                                unfocusedContainerColor = if (isDark) Color(0xFF2A3942) else Color.White,
                                disabledContainerColor = if (isDark) Color(0xFF222E35) else Color(0xFFE9ECEF),
                                focusedBorderColor = EmeraldPrimary,
                                unfocusedBorderColor = Color.Transparent
                            ),
                            singleLine = false,
                            maxLines = 4
                        )

                        Spacer(modifier = Modifier.width(8.dp))

                        val canSend = inputText.isNotBlank() && !isGenerating
                        Box(
                            modifier = Modifier
                                .size(48.dp)
                                .clip(CircleShape)
                                .background(
                                    if (canSend) EmeraldPrimary else Color.Gray.copy(alpha = 0.4f)
                                )
                                .clickable(enabled = canSend) {
                                    val text = inputText.trim()
                                    inputText = ""
                                    onSendMessage(text)
                                }
                                .testTag("send_button"),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Send,
                                contentDescription = "পাঠান",
                                tint = Color.White,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }
            }
        }
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(if (isDark) ChatBackgroundDark else ChatBackgroundLight)
                .padding(paddingValues)
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Material 3 Error Banner when an error occurs
                AnimatedVisibility(
                    visible = errorMessage != null,
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    if (errorMessage != null) {
                        Surface(
                            color = MaterialTheme.colorScheme.errorContainer,
                            modifier = Modifier.fillMaxWidth(),
                            shadowElevation = 2.dp
                        ) {
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Icon(
                                    imageVector = Icons.Default.ErrorOutline,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.onErrorContainer,
                                    modifier = Modifier.size(24.dp)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = "অন-ডিভাইস ইনফারেন্স সতর্কতা",
                                        style = MaterialTheme.typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                    Text(
                                        text = errorMessage,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                }
                                Spacer(modifier = Modifier.width(4.dp))
                                TextButton(onClick = { onToggleDiagnostics(true) }) {
                                    Text("ABI বিবরণ", fontSize = 12.sp)
                                }
                                IconButton(
                                    onClick = onDismissError,
                                    modifier = Modifier.size(32.dp)
                                ) {
                                    Icon(
                                        imageVector = Icons.Default.Close,
                                        contentDescription = "বন্ধ করুন",
                                        tint = MaterialTheme.colorScheme.onErrorContainer,
                                        modifier = Modifier.size(18.dp)
                                    )
                                }
                            }
                        }
                    }
                }

                // Chat Messages List
                if (messages.isEmpty()) {
                    EmptyChatGreeting(
                        samplePrompts = samplePrompts,
                        onSelectPrompt = { prompt -> onSendMessage(prompt) }
                    )
                } else {
                    LazyColumn(
                        state = listState,
                        modifier = Modifier.fillMaxSize(),
                        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(messages, key = { it.id }) { message ->
                            ChatMessageBubble(message = message, activeModelName = activePreset.shortName)
                        }

                        if (isGenerating) {
                            item {
                                TypingIndicatorBubble()
                            }
                        }
                    }
                }
            }
        }
    }

    if (showModelSwitcherDialog) {
        ModelSwitcherDialog(
            activePreset = activePreset,
            installedPresets = installedPresets,
            onDismiss = { onToggleModelSwitcher(false) },
            onSwitchModel = { preset ->
                onToggleModelSwitcher(false)
                onSwitchModel(preset)
            },
            onDownloadPreset = { preset ->
                onToggleModelSwitcher(false)
                onDownloadPreset(preset)
            },
            onDeletePreset = onDeletePreset,
            onImportModel = { uri ->
                onImportModel(uri)
                onToggleModelSwitcher(false)
            }
        )
    }

    if (showModelInfoDialog) {
        ModelInfoDialog(
            activePreset = activePreset,
            onDismiss = { onToggleModelInfo(false) }
        )
    }

    if (showDiagnosticsDialog) {
        HardwareDiagnosticsDialog(
            onDismiss = { onToggleDiagnostics(false) },
            onRedownload = {
                onToggleDiagnostics(false)
                onToggleModelSwitcher(true)
            }
        )
    }
}

/**
 * Dialog allowing user to switch between downloaded models or start downloading new presets.
 */
@Composable
fun ModelSwitcherDialog(
    activePreset: ModelDownloader.ModelPreset,
    installedPresets: List<ModelDownloader.ModelPreset>,
    onDismiss: () -> Unit,
    onSwitchModel: (ModelDownloader.ModelPreset) -> Unit,
    onDownloadPreset: (ModelDownloader.ModelPreset) -> Unit,
    onDeletePreset: (ModelDownloader.ModelPreset) -> Unit,
    onImportModel: (Uri) -> Unit
) {
    val uploadPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            onImportModel(uri)
        }
    }
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.Refresh,
                contentDescription = null,
                tint = EmeraldPrimary
            )
        },
        title = {
            Text(
                text = "মডেল নির্বাচন ও পরিবর্তন",
                fontWeight = FontWeight.Bold
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                // কাস্টম মডেল আপলোড কার্ড (Upload Custom Model)
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.45f),
                    border = BorderStroke(1.dp, EmeraldPrimary.copy(alpha = 0.6f)),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Memory,
                                contentDescription = null,
                                tint = EmeraldPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "নতুন GGUF মডেল আপলোড করুন",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onPrimaryContainer
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Colab থেকে তৈরি করা আপনার নতুন ৫০M GGUF ফাইলটি সরাসরি ফোন স্টোরেজ থেকে আপলোড করে সক্রিয় করুন।",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.85f)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Button(
                            onClick = { uploadPickerLauncher.launch("*/*") },
                            shape = RoundedCornerShape(8.dp),
                            colors = ButtonDefaults.buttonColors(containerColor = EmeraldPrimary),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Icon(imageVector = Icons.Default.CloudDownload, contentDescription = null, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("ফোন থেকে .gguf ফাইল নির্বাচন করুন", fontSize = 12.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = "অথবা নিচের সংরক্ষিত মডেলগুলো থেকে বেছে নিন:",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                ModelDownloader.PRESETS.forEach { preset ->
                    val isInstalled = installedPresets.any { it.id == preset.id }
                    val isActive = activePreset.id == preset.id

                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = if (isActive) EmeraldPrimary.copy(alpha = 0.12f) else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                        border = if (isActive) BorderStroke(1.5.dp, EmeraldPrimary) else null,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = preset.name,
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Bold,
                                    modifier = Modifier.weight(1f)
                                )
                                Surface(
                                    shape = RoundedCornerShape(6.dp),
                                    color = if (isActive) OnlineGreen.copy(alpha = 0.2f) else EmeraldPrimary.copy(alpha = 0.15f)
                                ) {
                                    Text(
                                        text = if (isActive) "সক্রিয়" else "~${preset.sizeMb} MB",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = if (isActive) OnlineGreen else EmeraldPrimary,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = preset.description,
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            Spacer(modifier = Modifier.height(8.dp))

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.End,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                if (isActive) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.CheckCircle,
                                            contentDescription = null,
                                            tint = OnlineGreen,
                                            modifier = Modifier.size(16.dp)
                                        )
                                        Text(
                                            text = "রানিং",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = OnlineGreen,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                } else if (isInstalled) {
                                    Button(
                                        onClick = { onSwitchModel(preset) },
                                        shape = RoundedCornerShape(8.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = EmeraldPrimary)
                                    ) {
                                        Text("সক্রিয় করুন", fontSize = 12.sp)
                                    }
                                    Spacer(modifier = Modifier.width(6.dp))
                                    IconButton(
                                        onClick = { onDeletePreset(preset) },
                                        modifier = Modifier.size(32.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Delete,
                                            contentDescription = "মুছুন",
                                            tint = MaterialTheme.colorScheme.error,
                                            modifier = Modifier.size(18.dp)
                                        )
                                    }
                                } else {
                                    Button(
                                        onClick = { onDownloadPreset(preset) },
                                        shape = RoundedCornerShape(8.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = EmeraldPrimary)
                                    ) {
                                        Icon(imageVector = Icons.Default.CloudDownload, contentDescription = null, modifier = Modifier.size(14.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text("ডাউনলোড", fontSize = 12.sp)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("বন্ধ করুন")
            }
        }
    )
}

/**
 * Message Bubble (User on right, AI on left) with timestamp and copy option.
 */
@Composable
fun ChatMessageBubble(message: ChatMessage, activeModelName: String = "বাংলা এআই") {
    val isDark = isSystemInDarkTheme()
    val isUser = message.isUser
    val timeFormat = remember { SimpleDateFormat("hh:mm a", Locale.getDefault()) }
    val formattedTime = remember(message.timestamp) { timeFormat.format(Date(message.timestamp)) }
    val context = LocalContext.current

    val bubbleColor = when {
        isUser && isDark -> UserBubbleDark
        isUser && !isDark -> UserBubbleLight
        !isUser && isDark -> AiBubbleDark
        else -> AiBubbleLight
    }

    val textColor = when {
        isUser && isDark -> Color.White
        isUser && !isDark -> Color(0xFF111B21)
        !isUser && isDark -> Color.White
        else -> Color(0xFF111B21)
    }

    val bubbleShape = if (isUser) {
        RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 2.dp)
    } else {
        RoundedCornerShape(topStart = 16.dp, topEnd = 16.dp, bottomStart = 2.dp, bottomEnd = 16.dp)
    }

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start
    ) {
        Card(
            shape = bubbleShape,
            colors = CardDefaults.cardColors(containerColor = bubbleColor),
            elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
            modifier = Modifier.widthIn(min = 60.dp, max = 330.dp)
        ) {
            Column(
                modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                if (!isUser) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        modifier = Modifier.padding(bottom = 3.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.SmartToy,
                            contentDescription = null,
                            modifier = Modifier.size(14.dp),
                            tint = EmeraldPrimary
                        )
                        Text(
                            text = "$activeModelName (GGUF)",
                            style = MaterialTheme.typography.labelSmall,
                            color = EmeraldPrimary,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Text(
                    text = if (message.text.isEmpty()) "..." else message.text,
                    style = MaterialTheme.typography.bodyMedium,
                    color = textColor,
                    lineHeight = 22.sp
                )

                Spacer(modifier = Modifier.height(4.dp))

                Row(
                    modifier = Modifier.align(Alignment.End),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    IconButton(
                        onClick = {
                            val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                            val clip = ClipData.newPlainText("Chat Message", message.text)
                            clipboard.setPrimaryClip(clip)
                        },
                        modifier = Modifier.size(18.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.ContentCopy,
                            contentDescription = "কপি করুন",
                            tint = textColor.copy(alpha = 0.5f),
                            modifier = Modifier.size(12.dp)
                        )
                    }

                    Text(
                        text = formattedTime,
                        fontSize = 10.sp,
                        color = textColor.copy(alpha = 0.6f)
                    )
                    if (isUser) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = "পাঠানো হয়েছে",
                            modifier = Modifier.size(12.dp),
                            tint = Color(0xFF53BDEB)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Animated 3-dot typing indicator when the model is generating.
 */
@Composable
fun TypingIndicatorBubble() {
    val isDark = isSystemInDarkTheme()
    val bubbleColor = if (isDark) AiBubbleDark else AiBubbleLight

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Start
    ) {
        Card(
            shape = RoundedCornerShape(16.dp, 16.dp, 16.dp, 2.dp),
            colors = CardDefaults.cardColors(containerColor = bubbleColor),
            elevation = CardDefaults.cardElevation(defaultElevation = 1.dp),
            modifier = Modifier.padding(vertical = 4.dp)
        ) {
            Row(
                modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "মডেল চিন্তা করছে",
                    style = MaterialTheme.typography.labelMedium,
                    color = EmeraldPrimary,
                    fontWeight = FontWeight.Medium
                )

                Row(
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    repeat(3) { index ->
                        val infiniteTransition = rememberInfiniteTransition(label = "dot_$index")
                        val scale by infiniteTransition.animateFloat(
                            initialValue = 0.4f,
                            targetValue = 1.0f,
                            animationSpec = infiniteRepeatable(
                                animation = tween(durationMillis = 600, delayMillis = index * 180),
                                repeatMode = RepeatMode.Reverse
                            ),
                            label = "scale"
                        )

                        Box(
                            modifier = Modifier
                                .size(6.dp * scale)
                                .clip(CircleShape)
                                .background(EmeraldPrimary)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Empty Chat state showing offline capability and sample questions.
 */
@Composable
fun EmptyChatGreeting(
    samplePrompts: List<String>,
    onSelectPrompt: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(72.dp)
                .clip(CircleShape)
                .background(EmeraldPrimary.copy(alpha = 0.15f)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Memory,
                contentDescription = null,
                tint = EmeraldPrimary,
                modifier = Modifier.size(36.dp)
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "বাংলা অফলাইন এআই",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(6.dp))

        Text(
            text = "১০০% অন-ডিভাইস LLM • সম্পূর্ণ ইন্টারনেট মুক্ত",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(24.dp))

        Text(
            text = "কিছু সাধারণ প্রশ্ন দিয়ে শুরু করতে পারেন:",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
            fontWeight = FontWeight.SemiBold
        )

        Spacer(modifier = Modifier.height(12.dp))

        Column(
            verticalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.fillMaxWidth(0.92f)
        ) {
            samplePrompts.forEach { prompt ->
                Surface(
                    shape = RoundedCornerShape(12.dp),
                    color = MaterialTheme.colorScheme.surface,
                    shadowElevation = 1.dp,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelectPrompt(prompt) }
                ) {
                    Text(
                        text = "💬 $prompt",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp)
                    )
                }
            }
        }
    }
}

/**
 * Modern Download Card with multi-preset selection, percentage progress bar,
 * download speed, downloaded MBs, custom URL input, local file picker, and cancel/retry options.
 */
@Composable
fun DownloadScreen(
    downloadState: ModelDownloader.DownloadState,
    activePreset: ModelDownloader.ModelPreset,
    installedPresets: List<ModelDownloader.ModelPreset>,
    onStartDownload: (ModelDownloader.ModelPreset, String?) -> Unit,
    onImportFromUri: (Uri, ModelDownloader.ModelPreset) -> Unit,
    onCancelDownload: () -> Unit,
    onShowDiagnostics: () -> Unit,
    onSwitchModel: (ModelDownloader.ModelPreset) -> Unit,
    modifier: Modifier = Modifier
) {
    val isDark = isSystemInDarkTheme()
    var selectedPreset by remember { mutableStateOf(activePreset) }
    var customUrl by remember { mutableStateOf(activePreset.url) }
    var showUrlEdit by remember { mutableStateOf(false) }

    LaunchedEffect(selectedPreset) {
        customUrl = selectedPreset.url
    }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri != null) {
            onImportFromUri(uri, selectedPreset)
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize()
    ) { paddingValues ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .background(if (isDark) ChatBackgroundDark else ChatBackgroundLight),
            contentAlignment = Alignment.Center
        ) {
            Card(
                modifier = Modifier
                    .fillMaxWidth(0.92f)
                    .padding(16.dp)
                    .verticalScroll(rememberScrollState()),
                shape = RoundedCornerShape(24.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
            ) {
                Column(
                    modifier = Modifier.padding(20.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    // Header Icon
                    Box(
                        modifier = Modifier
                            .size(60.dp)
                            .clip(CircleShape)
                            .background(EmeraldPrimary.copy(alpha = 0.12f)),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.CloudDownload,
                            contentDescription = null,
                            tint = EmeraldPrimary,
                            modifier = Modifier.size(32.dp)
                        )
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    Text(
                        text = "বাংলা এআই মডেল সেটআপ",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )

                    Spacer(modifier = Modifier.height(4.dp))

                    Text(
                        text = "itel A60 (ARM32) ও আধুনিক ফোনে অফলাইনে চালানোর জন্য যেকোনো একটি মডেল ডাউনলোড করুন।",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )

                    // If user already has another model installed, show option to jump back to it
                    if (installedPresets.isNotEmpty()) {
                        Spacer(modifier = Modifier.height(10.dp))
                        val firstInstalled = installedPresets.first()
                        OutlinedButton(
                            onClick = { onSwitchModel(firstInstalled) },
                            shape = RoundedCornerShape(10.dp)
                        ) {
                            Text("ডাউনলোড করা মডেলে ফিরে যান (${firstInstalled.shortName})")
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    // Model Preset Selection
                    Text(
                        text = "অন-ডিভাইস মডেল নির্বাচন করুন:",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.fillMaxWidth()
                    )
                    Spacer(modifier = Modifier.height(6.dp))

                    ModelDownloader.PRESETS.forEach { preset ->
                        val isSelected = selectedPreset.id == preset.id
                        val isInstalled = installedPresets.any { it.id == preset.id }

                        Surface(
                            onClick = {
                                selectedPreset = preset
                            },
                            shape = RoundedCornerShape(12.dp),
                            color = if (isSelected) EmeraldPrimary.copy(alpha = 0.12f) else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
                            border = if (isSelected) BorderStroke(1.5.dp, EmeraldPrimary) else null,
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                        ) {
                            Row(
                                modifier = Modifier.padding(12.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                RadioButton(
                                    selected = isSelected,
                                    onClick = { selectedPreset = preset },
                                    colors = RadioButtonDefaults.colors(selectedColor = EmeraldPrimary)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Column(modifier = Modifier.weight(1f)) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.SpaceBetween,
                                        modifier = Modifier.fillMaxWidth()
                                    ) {
                                        Text(
                                            text = preset.name,
                                            style = MaterialTheme.typography.bodyMedium,
                                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium
                                        )
                                        Surface(
                                            shape = RoundedCornerShape(6.dp),
                                            color = if (isInstalled) OnlineGreen.copy(alpha = 0.2f) else EmeraldPrimary.copy(alpha = 0.15f)
                                        ) {
                                            Text(
                                                text = if (isInstalled) "ডাউনলোড আছে" else "~${preset.sizeMb} MB",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = if (isInstalled) OnlineGreen else EmeraldPrimary,
                                                fontWeight = FontWeight.Bold,
                                                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                                            )
                                        }
                                    }
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text(
                                        text = preset.description,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    // Model Specs Summary Card
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(12.dp)) {
                            SpecRow("নির্বাচিত মডেল", selectedPreset.name)
                            SpecRow("আকার", "~${selectedPreset.sizeMb} MB (GGUF Quantized)")
                            SpecRow("ডিভাইস ABI", "${HardwareInfo.primaryAbi} (${if (HardwareInfo.is64Bit) "64-bit" else "32-bit armeabi-v7a"})")
                            SpecRow("রানটাইম ইঞ্জিন", "Native llama.cpp C++ JNI Engine")
                            SpecRow("অফলাইন স্ট্যাটাস", "১০০% অন-ডিভাইস")
                        }
                    }

                    Spacer(modifier = Modifier.height(16.dp))

                    when (downloadState) {
                        is ModelDownloader.DownloadState.Idle -> {
                            Column(
                                modifier = Modifier.fillMaxWidth(),
                                verticalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                Button(
                                    onClick = { onStartDownload(selectedPreset, if (showUrlEdit) customUrl else null) },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .testTag("download_button"),
                                    shape = RoundedCornerShape(12.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = EmeraldPrimary)
                                ) {
                                    Icon(imageVector = Icons.Default.CloudDownload, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "ডাউনলোড শুরু করুন (~${selectedPreset.sizeMb} MB)",
                                        fontWeight = FontWeight.Bold
                                    )
                                }

                                OutlinedButton(
                                    onClick = { filePickerLauncher.launch("*/*") },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .testTag("pick_local_file_button"),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.Memory, contentDescription = null)
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "ফোন স্টোরেজ থেকে GGUF নির্বাচন করুন",
                                        fontWeight = FontWeight.Medium
                                    )
                                }
                            }
                        }

                        is ModelDownloader.DownloadState.Downloading -> {
                            Column(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalAlignment = Alignment.CenterHorizontally
                            ) {
                                LinearProgressIndicator(
                                    progress = { downloadState.percentage / 100f },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(10.dp)
                                        .clip(RoundedCornerShape(5.dp)),
                                    color = EmeraldPrimary
                                )

                                Spacer(modifier = Modifier.height(10.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(
                                        verticalAlignment = Alignment.CenterVertically,
                                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Speed,
                                            contentDescription = null,
                                            modifier = Modifier.size(14.dp),
                                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                                        )
                                        Text(
                                            text = downloadState.speedFormatted,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            fontWeight = FontWeight.Medium
                                        )
                                    }

                                    Text(
                                        text = "${downloadState.percentage}%",
                                        style = MaterialTheme.typography.bodyMedium,
                                        fontWeight = FontWeight.Bold,
                                        color = EmeraldPrimary
                                    )
                                }

                                Spacer(modifier = Modifier.height(4.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = downloadState.downloadedFormatted,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                    Text(
                                        text = downloadState.statusBengali,
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.outline
                                    )
                                }

                                Spacer(modifier = Modifier.height(14.dp))

                                OutlinedButton(
                                    onClick = onCancelDownload,
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .testTag("cancel_download_button"),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.Cancel, contentDescription = null)
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("ডাউনলোড বাতিল করুন")
                                }
                            }
                        }

                        is ModelDownloader.DownloadState.Success -> {
                            Text(
                                text = "ডাউনলোড সফল! মেমরিতে মডেল লোড হচ্ছে...",
                                style = MaterialTheme.typography.bodyMedium,
                                color = OnlineGreen,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        is ModelDownloader.DownloadState.Error -> {
                            Column(
                                horizontalAlignment = Alignment.CenterHorizontally,
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Surface(
                                    shape = RoundedCornerShape(10.dp),
                                    color = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.8f),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Column(modifier = Modifier.padding(12.dp)) {
                                        Text(
                                            text = downloadState.message,
                                            style = MaterialTheme.typography.bodySmall,
                                            color = MaterialTheme.colorScheme.onErrorContainer,
                                            fontWeight = FontWeight.SemiBold
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "যদি নেটওয়ার্ক লিমিট বা CDN সমস্যা হয়, তবে আপনি ফোন স্টোরেজে .gguf ফাইল কপি করে সরাসরি নির্বাচন করতে পারেন অথবা বিকল্প URL দিতে পারেন।",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.85f)
                                        )
                                    }
                                }

                                Spacer(modifier = Modifier.height(14.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                                ) {
                                    Button(
                                        onClick = { onStartDownload(selectedPreset, if (showUrlEdit) customUrl else null) },
                                        modifier = Modifier
                                            .weight(1f)
                                            .testTag("retry_button"),
                                        shape = RoundedCornerShape(12.dp),
                                        colors = ButtonDefaults.buttonColors(containerColor = EmeraldPrimary)
                                    ) {
                                        Icon(imageVector = Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(16.dp))
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Text("পুনরায় চেষ্টা")
                                    }

                                    OutlinedButton(
                                        onClick = onCancelDownload,
                                        modifier = Modifier.weight(1f),
                                        shape = RoundedCornerShape(12.dp)
                                    ) {
                                        Text("বাতিল")
                                    }
                                }

                                Spacer(modifier = Modifier.height(8.dp))

                                OutlinedButton(
                                    onClick = { filePickerLauncher.launch("*/*") },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .testTag("pick_local_file_button"),
                                    shape = RoundedCornerShape(12.dp)
                                ) {
                                    Icon(imageVector = Icons.Default.Memory, contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text("ফোন স্টোরেজ থেকে GGUF নির্বাচন করুন")
                                }
                            }
                        }
                    }

                    Spacer(modifier = Modifier.height(14.dp))

                    // Optional Custom URL section
                    TextButton(
                        onClick = { showUrlEdit = !showUrlEdit }
                    ) {
                        Text(
                            text = if (showUrlEdit) "URL ইনপুট লুকান" else "কাস্টম ডাউনলোড URL পরিবর্তন করুন",
                            fontSize = 12.sp,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }

                    AnimatedVisibility(visible = showUrlEdit) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 8.dp)
                        ) {
                            OutlinedTextField(
                                value = customUrl,
                                onValueChange = { customUrl = it },
                                label = { Text("মডেল ডাউনলোড URL (GGUF)") },
                                placeholder = { Text("https://...") },
                                singleLine = false,
                                maxLines = 3,
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(12.dp)
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "ModelScope, Hugging Face, GitHub Releases বা যেকোনো সরাসরি HTTP/HTTPS ডাউনলোড লিংক এখানে দিতে পারেন।",
                                style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.outline
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(10.dp))

                    TextButton(onClick = onShowDiagnostics) {
                        Icon(imageVector = Icons.Default.DeveloperMode, contentDescription = null, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text("হার্ডওয়্যার ও ABI ডায়াগনস্টিকস", fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun SpecRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 2.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            text = value,
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold
        )
    }
}

/**
 * Model specifications dialog showing hyperparameters and architecture.
 */
@Composable
fun ModelInfoDialog(
    activePreset: ModelDownloader.ModelPreset,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.Memory,
                contentDescription = null,
                tint = EmeraldPrimary
            )
        },
        title = {
            Text(
                text = "মডেলের প্রযুক্তিগত বিবরণ",
                fontWeight = FontWeight.Bold
            )
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(6.dp),
                modifier = Modifier.verticalScroll(rememberScrollState())
            ) {
                Text(
                    text = "এই মডেলটি সরাসরি আপনার ফোনের RAM এবং CPU ব্যবহার করে অফলাইনে চলে।",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(modifier = Modifier.height(6.dp))
                SpecRow("মডেলের নাম", activePreset.name)
                SpecRow("ফাইল ফরম্যাট", "GGUF Quantized")
                SpecRow("আনুমানিক আকার", "~${activePreset.sizeMb} MB")
                SpecRow("Max New Tokens", "${BengaliGptEngine.MAX_NEW_TOKENS}")
                SpecRow("Temperature", "${BengaliGptEngine.TEMPERATURE}")
                SpecRow("Top-P", "${BengaliGptEngine.TOP_P}")
                SpecRow("Repetition Penalty", "${BengaliGptEngine.REPETITION_PENALTY}")
                SpecRow("ইঞ্জিন রানটাইম", "llama.cpp GGML C++ JNI Engine")
                SpecRow("32-bit & 64-bit", "armeabi-v7a ও arm64-v8a নেটিভ সাপোর্ট")
                SpecRow("সংযোগ", "১০০% অফলাইন (অন-ডিভাইস)")
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("ঠিক আছে")
            }
        }
    )
}

/**
 * Hardware and Architecture Diagnostics Dialog for 32-bit armeabi-v7a vs 64-bit arm64-v8a.
 */
@Composable
fun HardwareDiagnosticsDialog(
    onDismiss: () -> Unit,
    onRedownload: () -> Unit
) {
    val context = LocalContext.current
    val memoryInfo = remember { HardwareInfo.getMemoryInfo(context) }
    val is32Bit = !HardwareInfo.is64Bit

    AlertDialog(
        onDismissRequest = onDismiss,
        icon = {
            Icon(
                imageVector = Icons.Default.DeveloperMode,
                contentDescription = null,
                tint = EmeraldPrimary
            )
        },
        title = {
            Text(
                text = "হার্ডওয়্যার ও ABI ডায়াগনস্টিকস",
                fontWeight = FontWeight.Bold
            )
        },
        text = {
            Column(
                verticalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.verticalScroll(rememberScrollState())
            ) {
                SpecRow("ডিভাইস মডেল", HardwareInfo.deviceModel)
                SpecRow("অ্যান্ড্রয়েড সংস্করণ", HardwareInfo.androidVersion)
                SpecRow("প্রাইমারি ABI", HardwareInfo.primaryAbi)
                SpecRow("সকল সাপোর্ট ABI", HardwareInfo.supportedAbis.joinToString(", "))
                SpecRow("আর্কিটেকচার", if (is32Bit) "32-bit (armeabi-v7a)" else "64-bit (arm64-v8a)")
                SpecRow("ডিভাইস RAM", "${HardwareInfo.formatBytes(memoryInfo.first)} ফ্রি / ${HardwareInfo.formatBytes(memoryInfo.second)} মোট")

                Spacer(modifier = Modifier.height(8.dp))

                Surface(
                    shape = RoundedCornerShape(8.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(10.dp)) {
                        Text(
                            text = "32-bit ও 64-bit আর্কিটেকচার সাপোর্ট:",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Native llama.cpp C++ ইঞ্জিনটি 32-bit (armeabi-v7a - যেমন itel A60 / Unisoc) এবং 64-bit (arm64-v8a) উভয় আর্কিটেকচারের জন্যই অপ্টিমাইজড কম্পাইল করা আছে।",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("বন্ধ করুন")
            }
        },
        dismissButton = {
            TextButton(onClick = {
                onDismiss()
                onRedownload()
            }) {
                Text("মডেল পরিবর্তন")
            }
        }
    )
}
