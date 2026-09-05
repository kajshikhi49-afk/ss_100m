"""
Text generation and Proactive Interactive Conversation pipeline for 10M Bengali GPT model.
Supports both standard generation and Socratic proactive follow-up dialogue.
"""

import os
import torch
from src.config import GPTConfig
from src.model import BengaliGPT
from src.tokenizer import BengaliTokenizer
from src.lora import apply_lora, load_lora


def load_trained_model(base_checkpoint=None, lora_checkpoint=None, tokenizer_path="tokenizer.json", device=None):
    """Load model and tokenizer with checkpoint support."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BengaliTokenizer(vocab_size=10000)
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    elif os.path.exists("checkpoints/tokenizer.json"):
        tokenizer.load("checkpoints/tokenizer.json")
    else:
        tokenizer.build_vocab("data/corpus.txt")
        tokenizer.save("tokenizer.json")

    config = GPTConfig()
    config.vocab_size = tokenizer.vocab_size
    config.device = device

    model = BengaliGPT(config)

    if base_checkpoint and os.path.exists(base_checkpoint):
        print(f"📥 Loading base model from {base_checkpoint}...")
        model.load_state_dict(torch.load(base_checkpoint, map_location='cpu'))

    if lora_checkpoint and os.path.exists(lora_checkpoint):
        print(f"📥 Loading LoRA adapter from {lora_checkpoint}...")
        model, _ = apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha, target_modules=config.lora_target_modules)
        load_lora(model, lora_checkpoint)

    model.to(device)
    model.eval()
    return model, tokenizer, config


def generate_text(prompt="বাংলাদেশের রাজধানী",
                  base_checkpoint=None,
                  lora_checkpoint=None,
                  tokenizer_path="tokenizer.json",
                  max_new_tokens=150,
                  temperature=0.8,
                  top_k=50,
                  device=None):
    """Autoregressive text generation."""
    model, tokenizer, config = load_trained_model(base_checkpoint, lora_checkpoint, tokenizer_path, device)

    input_ids = tokenizer.encode(prompt, add_bos=True)
    idx = torch.tensor([input_ids], dtype=torch.long, device=config.device)

    print(f"\n🎯 Prompt: \"{prompt}\"")
    print(f"⚙️ Generating {max_new_tokens} tokens (Temp: {temperature}, Top-K: {top_k})...\n")

    out_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    generated_text = tokenizer.decode(out_idx[0].tolist(), skip_special_tokens=True)

    print("—" * 60)
    print(generated_text)
    print("—" * 60)
    return generated_text


def chat_interactive(base_checkpoint=None, lora_checkpoint=None, tokenizer_path="tokenizer.json"):
    """
    Proactive, conversational chat mode where the model answers questions,
    anticipates user intent, and proactively asks relevant follow-up questions.
    """
    model, tokenizer, config = load_trained_model(base_checkpoint, lora_checkpoint, tokenizer_path)

    system_prompt = (
        "<|system|>\n"
        "তুমি একজন অত্যন্ত বুদ্ধিমান, সহযোগী ও অনুসন্ধিৎসু বাংলা সহকারী। "
        "ব্যবহারকারীর প্রশ্নের সঠিক ও সুন্দর উত্তর দাও, এবং উত্তর দেওয়ার পর ব্যবহারকারীর কৌতূহল বুঝে "
        "তাকে পরবর্তী বিষয় জানার জন্য একটি প্রাসঙ্গিক প্রশ্ন (Follow-up Ask) করো।\n"
    )

    print("=" * 65)
    print("💬 ইন্টারেক্টিভ প্রোঅ্যাক্টিভ বাংলা চ্যাটবট (Interactive Socratic Chat)")
    print("মডেলটি আপনার প্রশ্নের উত্তর দেওয়ার পর নিজে থেকেই ফলো-আপ প্রশ্ন করবে!")
    print("বের হতে চাইলে 'exit' অথবা 'quit' লিখুন।")
    print("=" * 65)

    history = system_prompt

    while True:
        try:
            user_input = input("\n👤 আপনি: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'বাই', 'বিদায়']:
                print("🤖 সহকারী: ধন্যবাদ! আপনার সাথে কথা বলে ভালো লাগলো। আবার দেখা হবে!")
                break

            # Append user turn
            history += f"<|user|>\n{user_input}\n<|assistant|>\n"

            # Encode conversation history
            input_ids = tokenizer.encode(history, add_bos=True)
            # Crop to block_size if conversation is very long
            if len(input_ids) > config.block_size - 150:
                input_ids = input_ids[-(config.block_size - 150):]

            idx = torch.tensor([input_ids], dtype=torch.long, device=config.device)

            out_idx = model.generate(idx, max_new_tokens=150, temperature=0.75, top_k=40)
            generated_ids = out_idx[0][len(input_ids):].tolist()
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Clean any trailing special markers
            if "<|user|>" in response:
                response = response.split("<|user|>")[0].strip()

            print(f"🤖 সহকারী: {response}")

            # Keep conversation in history
            history += f"{response}\n"

        except (KeyboardInterrupt, EOFError):
            print("\n\nচ্যাট শেষ করা হলো।")
            break


if __name__ == "__main__":
    generate_text("বাংলা ভাষার ইতিহাস")
