# ComfyUI-Simple-LlamaCPP-Client

A lightweight custom node for **ComfyUI** that connects directly to a local **llama.cpp OpenAI-compatible server**.

It supports:

- 💬 Chat completions (`/v1/chat/completions`)
- 🖼 Vision models (image + prompt input)
- ⚡ Streaming mode (SSE token accumulation)
- 🧠 Clean Answer + Thinking separation
- 📦 Optional JSON-only output mode
- 🔑 Optional API key authentication
- 🎛 Minimal design (server-side parameters stay server-side)

---

## ✨ Features

✅ **Simple llama.cpp client inside ComfyUI**  
✅ Works with any OpenAI-compatible llama.cpp server  
✅ Supports **system prompt + user prompt**  
✅ Optional **image input** for multimodal models  
✅ Outputs:

- Answer (clean)
- Thinking (if available)
- JSON (if enabled)
- Raw server response
- Model used

✅ Auto-detects model from `/v1/models`  
✅ Optional override for max_tokens, seed, stop  
✅ Designed to match other Simple-* nodes

---

## 📦 Installation

### 1. Manual Install

Clone this repo into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ai-joe-git/ComfyUI-Simple-LlamaCPP-Client.git
```

Then restart ComfyUI.

---

### 2. Install via ComfyUI Manager

Once indexed, you can install directly through:

**ComfyUI Manager → Custom Nodes → Search → Simple LlamaCPP Client**

---

## 🚀 Server Requirements

This node expects a llama.cpp server running with OpenAI API compatibility:

### Example llama.cpp launch:

```bat
llama-server.exe ^
  -m Ministral-3-8B-Instruct.gguf ^
  --host 127.0.0.1 ^
  --port 8082 ^
  --mmproj mmproj.gguf ^
  -c 8192
```

The node connects to:

```
http://127.0.0.1:8082/v1/chat/completions
```

---

## 🧩 Node Inputs

| Input | Description |
|------|------------|
| `server_url` | llama.cpp server URL (default: `http://127.0.0.1:8082`) |
| `prompt` | User message text |
| `system_prompt` | Optional system instruction |
| `image` | Optional ComfyUI IMAGE input (vision models) |
| `api_key` | Optional Bearer token |
| `model_override` | Force a model name instead of autodetect |
| `stream` | Enable SSE streaming mode |
| `timeout_seconds` | Request timeout |
| `json_mode` | Force JSON-only output |
| `json_schema_hint` | Optional schema hint text |
| `max_tokens` | Optional override |
| `seed` | Optional override |
| `stop` | Optional stop sequence |

---

## 📤 Node Outputs

| Output | Description |
|-------|------------|
| `answer` | Final cleaned answer text |
| `thinking` | Internal reasoning if returned (`reasoning` or `<think>`) |
| `json` | Parsed JSON output (only if json_mode enabled) |
| `raw` | Full raw server response (debugging) |
| `model_used` | Model name used for the request |

---

## 🖼 Vision Support

If your llama.cpp server is started with:

```bash
--mmproj mmproj.gguf
```

Then you can connect an image directly into the node:

```
Load Image → Simple LlamaCPP Client → Text Output
```

The node sends:

- Text prompt
- Image as base64 PNG
- OpenAI-compatible vision message format

---

## 📦 JSON Mode Example

Enable:

- `json_mode = true`

Prompt:

```text
Return a JSON object with keys: title, mood, tags.
```

Output:

```json
{
  "title": "Night Convenience Store",
  "mood": "eerie",
  "tags": ["fluorescent", "adult swim", "retro"]
}
```

The node also includes fallback JSON extraction if the model adds extra text.

---

## 🔑 API Key Support

If your server is launched with:

```bash
--api-key opencode
```

Then simply enter in node:

```
api_key = opencode
```

If blank, no Authorization header is sent.

---

## 🛠 Troubleshooting

### ❌ 401 Unauthorized

Your server requires an API key.

Fix:

- Start llama.cpp without `--api-key`
- OR provide the correct key in the node

---

### ❌ No model detected

If `/v1/models` is unavailable, the node uses:

```
local-model
```

You can manually set:

```
model_override = your-model-name
```

---

### ❌ Thinking output is empty

Most llama.cpp models do not return reasoning unless:

- The model supports reasoning fields
- Or it outputs `<think>...</think>`

This is normal.

---

## 📜 License

MIT License (same style as other Simple-* repos).

---

## ⭐ Related Projects

- **Simple File Batcher**  
  https://github.com/ai-joe-git/ComfyUI-Simple-File-Batcher

- **Simple Prompt Batcher**  
  https://github.com/ai-joe-git/ComfyUI-Simple-Prompt-Batcher

---

## ❤️ Credits

Built for the ComfyUI community with a focus on:

- simplicity
- speed
- clean UX
- local-first AI workflows

Enjoy!
