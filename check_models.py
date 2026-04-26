import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import dotenv_values

def load_hf_token():
    # Try current dir .env files
    for p in Path('.').glob('.env*'):
        env = dotenv_values(str(p))
        if env.get('HF_TOKEN'):
            return env['HF_TOKEN']
    # Try os.environ
    return os.environ.get('HF_TOKEN')

def check_model(client, model_id):
    print(f"Checking model: {model_id}...", end="", flush=True)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
            max_tokens=10,
            timeout=30
        )
        content = response.choices[0].message.content.strip()
        print(f" SUCCESS: {content}")
        return True
    except Exception as e:
        print(f" FAILED: {e}")
        return False

def main():
    token = load_hf_token()
    if not token:
        print("ERROR: HF_TOKEN not found in .env files or os.environ")
        sys.exit(1)

    # Use the Hugging Face Router endpoint
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token
    )

    models = [
        "deepseek-ai/DeepSeek-V3",
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "google/gemma-3-27b-it"
    ]

    results = {}
    for m in models:
        results[m] = check_model(client, m)

    print("\nSummary:")
    for m, success in results.items():
        status = "WORKING" if success else "NOT WORKING"
        print(f"  {m}: {status}")

if __name__ == "__main__":
    main()
