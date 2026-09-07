package com.example

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExampleUnitTest {
  @Test
  fun addition_isCorrect() {
    assertEquals(4, 2 + 2)
  }

  @Test
  fun onnxEngine_formatPrompt_wrapsCorrectly() {
    val engine = OnnxEngine()
    val formatted = engine.formatPrompt("বাংলাদেশ কেমন দেশ?")
    assertEquals("প্রশ্ন: বাংলাদেশ কেমন দেশ? উত্তর:", formatted)
  }

  @Test
  fun onnxEngine_cleanResponse_stripsEOS() {
    val engine = OnnxEngine()
    val cleaned = engine.cleanResponse("উত্তর: এটি একটি সুন্দর দেশ。<EOS> কিছু অতিরিক্ত টেক্সট")
    assertEquals("এটি একটি সুন্দর দেশ。", cleaned)
  }
}
