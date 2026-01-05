# TuneKit

**Fine-tune SLMs in 15 minutes. No setup. No cost. Just results.**

<p align="center">
  <a href="https://tunekit.app"><strong>Try it now at tunekit.app</strong></a>
  <br />
  <sub>If you find this useful, please star the repo!</sub>
</p>

---

## What is TuneKit?

TuneKit turns your data into a fine-tuned model with zero infrastructure headaches. Upload your dataset, pick a model, and get a ready-to-run Colab notebook. That's it.

## How it works

```
Upload Data → Choose Model → Get Notebook → Run in Colab → Done
```

1. **Upload** your conversation data (JSONL)
2. **Select** from Llama, Phi, Gemma, Qwen, or Mistral
3. **Download** a pre-configured Colab notebook
4. **Run** on Google's free T4 GPU
5. **Export** your model (LoRA, GGUF, or merged)

## Why TuneKit?

| Feature | TuneKit | DIY |
|---------|---------|-----|
| Setup time | 0 min | Hours |
| Cost | $0 | $$ |
| GPU required | No (uses Colab) | Yes |
| Code to write | None | Lots |

**Powered by [Unsloth](https://github.com/unslothai/unsloth)** — 2x faster training, 70% less VRAM.

## Supported Models

- **Llama 3.2** (1B, 3B)
- **Phi-4 Mini**
- **Gemma 3** (1B, 4B)
- **Qwen 2.5** (1.5B, 3B)
- **Mistral 7B**

## Quick Start

1. Go to [tunekit.app](https://tunekit.app)
2. Upload your dataset
3. Click through the wizard
4. Open in Colab and hit "Run All"
5. Your fine-tuned model is ready in ~15 minutes

## Data Format

TuneKit expects conversation data in this format:

```json
{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}
{"messages": [{"role": "user", "content": "Help me"}, {"role": "assistant", "content": "Sure!"}]}
```

## Run Locally

```bash
git clone https://github.com/riyanshibohra/TuneKit.git
cd TuneKit
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open `http://localhost:8000` in your browser.

## License

MIT

---

<p align="center">
  <a href="https://tunekit.app">tunekit.app</a>
</p>
