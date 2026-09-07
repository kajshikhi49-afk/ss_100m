package com.example

import java.util.UUID

/**
 * Data model for a single chat message in the conversation history.
 */
data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val text: String,
    val isUser: Boolean,
    val timestamp: Long = System.currentTimeMillis()
)
