#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <atomic>
#include "llama.h"

#define TAG "BengaliAiJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model   * g_model   = nullptr;
static llama_context * g_context = nullptr;
static std::mutex      g_mutex;
static std::atomic<bool> g_cancel_flag(false);

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_BengaliGptEngine_nativeLoadModel(JNIEnv * env, jobject /* this */, jstring model_path, jint n_threads) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }

    const char * path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model from: %s", path);

    // Initialize backend
    llama_backend_init();

    // Model parameters
    llama_model_params mparams = llama_model_default_params();

    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(model_path, path);

    if (!g_model) {
        LOGE("Failed to load model from path!");
        return JNI_FALSE;
    }

    // Context parameters
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = n_threads > 0 ? n_threads : 4;
    cparams.n_threads_batch = n_threads > 0 ? n_threads : 4;

    g_context = llama_init_from_model(g_model, cparams);
    if (!g_context) {
        LOGE("Failed to create llama context!");
        llama_model_free(g_model);
        g_model = nullptr;
        return JNI_FALSE;
    }

    LOGI("Model and context initialized successfully!");
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_example_BengaliGptEngine_nativeCancelGeneration(JNIEnv * /* env */, jobject /* this */) {
    g_cancel_flag.store(true);
}

JNIEXPORT void JNICALL
Java_com_example_BengaliGptEngine_nativeGenerate(
        JNIEnv * env,
        jobject /* this */,
        jstring prompt_str,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat repetition_penalty,
        jobject callback
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_context) {
        LOGE("Model or context not loaded!");
        return;
    }

    g_cancel_flag.store(false);

    jclass cbClass = env->GetObjectClass(callback);
    jmethodID onTokenMethod = env->GetMethodID(cbClass, "onTokenBytes", "([B)Z");
    if (!onTokenMethod) {
        LOGE("Callback onTokenBytes method not found!");
        return;
    }

    const char * prompt = env->GetStringUTFChars(prompt_str, nullptr);
    const struct llama_vocab * vocab = llama_model_get_vocab(g_model);

    // Tokenize prompt
    const int n_prompt_max = 2048;
    std::vector<llama_token> prompt_tokens(n_prompt_max);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), prompt_tokens.data(), n_prompt_max, true, true);
    env->ReleaseStringUTFChars(prompt_str, prompt);

    if (n_tokens < 0) {
        LOGE("Tokenization failed!");
        return;
    }
    prompt_tokens.resize(n_tokens);
    LOGI("Prompt tokenized into %d tokens", n_tokens);

    // Setup sampler chain
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (repetition_penalty > 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), 64, repetition_penalty, 0.0f, 0.0f
        ));
    }
    if (top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.85f, 1));
    if (temperature > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    }

    // Clear KV cache for new prompt
    llama_memory_clear(llama_get_memory(g_context), true);

    // Evaluate prompt tokens in batch
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_tokens - 1);
        batch.n_tokens++;

        if (batch.n_tokens == 512 || i == n_tokens - 1) {
            if (llama_decode(g_context, batch) != 0) {
                LOGE("llama_decode failed on prompt!");
                llama_batch_free(batch);
                llama_sampler_free(smpl);
                return;
            }
            batch.n_tokens = 0;
        }
    }

    int n_cur = n_tokens;
    int n_gen = 0;
    char piece_buf[256];
    std::string accumulated;

    // Generation loop
    while (n_gen < max_tokens && !g_cancel_flag.load()) {
        llama_token new_token_id = llama_sampler_sample(smpl, g_context, -1);
        llama_sampler_accept(smpl, new_token_id);

        // Check for end of generation token
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            LOGI("EOG token reached");
            break;
        }

        // Convert token to text piece
        int piece_len = llama_token_to_piece(vocab, new_token_id, piece_buf, sizeof(piece_buf), 0, false);
        if (piece_len > 0) {
            accumulated.append(piece_buf, piece_len);

            // Check for multi-token stop sequences in output
            if (accumulated.find("\nপ্রশ্ন:") != std::string::npos ||
                accumulated.find("প্রশ্ন:") != std::string::npos ||
                accumulated.find("<EOS>") != std::string::npos ||
                accumulated.find("<|im_end|>") != std::string::npos ||
                accumulated.find("<|endoftext|>") != std::string::npos ||
                accumulated.find("<|im_start|>") != std::string::npos ||
                accumulated.find("\nuser\n") != std::string::npos ||
                accumulated.find("\nUser:") != std::string::npos) {
                LOGI("Stop sequence detected in output!");
                break;
            }

            jbyteArray jbytes = env->NewByteArray(piece_len);
            env->SetByteArrayRegion(jbytes, 0, piece_len, (const jbyte *)piece_buf);
            jboolean cont = env->CallBooleanMethod(callback, onTokenMethod, jbytes);
            env->DeleteLocalRef(jbytes);
            if (!cont) {
                LOGI("Generation stopped by callback");
                break;
            }
        }

        // Decode next single token
        batch.n_tokens = 0;
        batch.token[batch.n_tokens] = new_token_id;
        batch.pos[batch.n_tokens] = n_cur;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens = 1;

        n_cur++;
        n_gen++;

        if (llama_decode(g_context, batch) != 0) {
            LOGE("llama_decode failed during generation!");
            break;
        }
    }

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    LOGI("Generation finished. Tokens produced: %d", n_gen);
}

JNIEXPORT void JNICALL
Java_com_example_BengaliGptEngine_nativeClose(JNIEnv * /* env */, jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_context) {
        llama_free(g_context);
        g_context = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    llama_backend_free();
    LOGI("Model and backend closed.");
}

} // extern "C"
