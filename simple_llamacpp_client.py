import base64
import io
import json
import re
from typing import Any, Dict, Optional, Tuple, List

import requests
from PIL import Image

try:
    import torch
except Exception:
    torch = None

# Optional: ComfyUI progress (safe if missing)
try:
    import comfy.utils
except Exception:
    comfy = None


# -----------------------------
# Helpers
# -----------------------------

_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _fix_mojibake(text: str) -> str:
    """
    Fix common UTF-8/Latin1 mojibake like: Hereâs -> Here’s
    """
    if not text:
        return ""
    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def _replace_smart_quotes(text: str) -> str:
    """
    Replace curly quotes/apostrophes with straight ASCII quotes.
    """
    if not text:
        return ""
    return (
        text.replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
    )


def _postprocess_text(text: str, mode: str) -> str:
    """
    mode options:
      - "none"
      - "fix_mojibake"
      - "ascii_quotes"
      - "fix_mojibake+ascii_quotes"
    """
    t = text or ""
    mode = (mode or "fix_mojibake").strip()

    if mode == "none":
        return t
    if mode == "fix_mojibake":
        return _fix_mojibake(t)
    if mode == "ascii_quotes":
        return _replace_smart_quotes(t)
    if mode == "fix_mojibake+ascii_quotes":
        return _replace_smart_quotes(_fix_mojibake(t))

    return _fix_mojibake(t)


def _clean_answer(text: str) -> str:
    """Light cleanup of common wrappers without being destructive."""
    if not text:
        return ""
    t = str(text).strip()
    t = re.sub(r"^\s*(final|answer)\s*:\s*", "", t, flags=re.IGNORECASE)
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`").strip()
    return t.strip()


def _split_thinking_and_answer_from_content(content: str) -> Tuple[str, str]:
    """Extract <think>...</think> blocks if present."""
    if not content:
        return "", ""
    m = _THINK_TAG_RE.search(content)
    if m:
        thinking = (m.group(1) or "").strip()
        answer = _THINK_TAG_RE.sub("", content).strip()
        return thinking, _clean_answer(answer)
    return "", _clean_answer(content)


def _split_thinking_and_answer(message_obj: Dict[str, Any]) -> Tuple[str, str]:
    """
    Supports:
      - message.reasoning (preferred)
      - <think> tags in message.content
    """
    reasoning = ""
    content = ""

    if isinstance(message_obj, dict):
        reasoning = message_obj.get("reasoning") or message_obj.get("thoughts") or ""
        content = message_obj.get("content") or ""

    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip(), _clean_answer(content)

    if isinstance(content, str):
        return _split_thinking_and_answer_from_content(content)

    return "", _clean_answer(str(content))


def _image_tensor_to_base64_png(image_tensor) -> Optional[str]:
    """
    ComfyUI IMAGE is typically torch tensor [B,H,W,C] float 0..1.
    Convert first batch image to PNG base64 data URL.
    """
    if image_tensor is None:
        return None
    if torch is None:
        raise RuntimeError("torch not available but IMAGE was provided.")
    if len(image_tensor.shape) != 4:
        raise ValueError(f"Unexpected IMAGE tensor shape: {image_tensor.shape}")

    img = image_tensor[0].detach().cpu()

    # Robust scaling
    if img.max() <= 1.5:
        img = (img * 255.0).clamp(0, 255)
    else:
        img = img.clamp(0, 255)

    img = img.to(torch.uint8).numpy()
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    k = (api_key or "").strip()
    if k:
        headers["Authorization"] = f"Bearer {k}"
    return headers


def _safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_from_text(text: str) -> Optional[Any]:
    """
    Attempts to extract JSON object/array from a messy response:
    - strips code fences
    - finds first {...} or [...] block heuristically
    """
    if not text:
        return None
    t = text.strip()

    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    direct = _safe_json_loads(t)
    if direct is not None:
        return direct

    obj_start = t.find("{")
    arr_start = t.find("[")
    if obj_start == -1 and arr_start == -1:
        return None

    start = obj_start if (obj_start != -1 and (arr_start == -1 or obj_start < arr_start)) else arr_start
    candidate = t[start:].strip()

    for i in range(0, min(4000, len(candidate))):
        chunk = candidate[: len(candidate) - i].strip()
        if not chunk:
            break
        parsed = _safe_json_loads(chunk)
        if parsed is not None:
            return parsed

    return None


def _pick_model_from_models_payload(payload: Dict[str, Any]) -> Optional[str]:
    data = payload.get("data")
    if isinstance(data, list) and data:
        for item in data:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("name") or item.get("model")
                if isinstance(mid, str) and mid.strip():
                    return mid.strip()
    models = payload.get("models")
    if isinstance(models, list) and models:
        for item in models:
            if isinstance(item, dict):
                mid = item.get("id") or item.get("name") or item.get("model")
                if isinstance(mid, str) and mid.strip():
                    return mid.strip()
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _get_first_model_id(server_url: str, headers: Dict[str, str], timeout_s: int) -> Optional[str]:
    url = server_url.rstrip("/") + "/v1/models"
    try:
        r = requests.get(url, headers=headers, timeout=timeout_s)
        r.encoding = "utf-8"
        if r.status_code != 200:
            return None
        j = r.json()
        return _pick_model_from_models_payload(j)
    except Exception:
        return None


def _make_progress():
    if comfy is None:
        return None
    try:
        pb = comfy.utils.ProgressBar(1000)
        state = {"i": 0}

        def tick():
            state["i"] = min(state["i"] + 1, 1000)
            pb.update_absolute(state["i"])

        return tick
    except Exception:
        return None


# -----------------------------
# Streaming parser
# -----------------------------

def _iter_sse_lines(resp):
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        yield line


def _parse_stream_and_accumulate(resp, tick=None) -> Tuple[str, str, Dict[str, Any]]:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    meta = {"done": False, "chunks": 0}

    for line in _iter_sse_lines(resp):
        if tick:
            tick()

        data_str = line[len("data:"):].strip() if line.startswith("data:") else line.strip()

        if data_str == "[DONE]":
            meta["done"] = True
            break

        chunk = _safe_json_loads(data_str)
        if not isinstance(chunk, dict):
            continue

        meta["chunks"] += 1

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        c0 = choices[0] if isinstance(choices[0], dict) else {}

        delta = c0.get("delta") if isinstance(c0, dict) else None
        if isinstance(delta, dict):
            d_content = delta.get("content")
            if isinstance(d_content, str) and d_content:
                content_parts.append(d_content)

            d_reason = delta.get("reasoning") or delta.get("thoughts")
            if isinstance(d_reason, str) and d_reason:
                reasoning_parts.append(d_reason)

        msg = c0.get("message")
        if isinstance(msg, dict):
            m_content = msg.get("content")
            if isinstance(m_content, str) and m_content:
                content_parts.append(m_content)
            m_reason = msg.get("reasoning") or msg.get("thoughts")
            if isinstance(m_reason, str) and m_reason:
                reasoning_parts.append(m_reason)

    full_content = "".join(content_parts).strip()
    full_reasoning = "".join(reasoning_parts).strip()

    if not full_reasoning and full_content:
        think, ans = _split_thinking_and_answer_from_content(full_content)
        if think:
            full_reasoning = think
            full_content = ans

    return full_content, full_reasoning, meta


# -----------------------------
# Node
# -----------------------------

class SimpleLlamaCppClient:
    """
    ComfyUI node: OpenAI-compatible llama.cpp client (chat completions).
    - UTF-8 decoding enforced
    - Dropdowns for postprocess / model mode / stop mode
    """

    @classmethod
    def INPUT_TYPES(cls):
        # IMPORTANT: In ComfyUI, dropdowns are created by using a LIST as the type,
        # not by "STRING + choices". (That's why your UI showed text inputs.)
        return {
            "required": {
                "server_url": ("STRING", {"default": "http://127.0.0.1:8082"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),

                "api_key": ("STRING", {"default": ""}),

                # Model selection: dropdown + optional custom text
                "model_mode": (["auto", "custom"], {"default": "auto"}),
                "model_override": ("STRING", {"default": ""}),  # used only if model_mode == custom

                "stream": ("BOOLEAN", {"default": True}),
                "timeout_seconds": ("INT", {"default": 300, "min": 1, "max": 3600}),

                "json_mode": ("BOOLEAN", {"default": False}),
                "json_schema_hint": ("STRING", {"multiline": True, "default": ""}),

                "max_tokens": ("INT", {"default": 0, "min": 0, "max": 131072}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),

                # Stop: dropdown presets + optional custom
                "stop_mode": (["none", "preset:common_eot", "preset:triple_hash", "custom"], {"default": "none"}),
                "stop_custom": ("STRING", {"default": ""}),

                # Text post-process dropdown
                "text_postprocess": (["fix_mojibake", "none", "ascii_quotes", "fix_mojibake+ascii_quotes"],
                                     {"default": "fix_mojibake"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("answer", "thinking", "json", "raw", "model_used")
    FUNCTION = "run"
    CATEGORY = "LLM / Simple llama.cpp"

    def run(
        self,
        server_url: str,
        prompt: str,
        system_prompt: str = "",
        image=None,
        api_key: str = "",
        model_mode: str = "auto",
        model_override: str = "",
        stream: bool = True,
        timeout_seconds: int = 300,
        json_mode: bool = False,
        json_schema_hint: str = "",
        max_tokens: int = 0,
        seed: int = -1,
        stop_mode: str = "none",
        stop_custom: str = "",
        text_postprocess: str = "fix_mojibake",
    ):
        server_url = (server_url or "").strip().rstrip("/")
        if not server_url:
            raise ValueError("server_url is empty")

        user_text = (prompt or "").strip()
        if not user_text:
            return ("", "", "", "", "")

        headers = _build_headers(api_key)
        timeout_s = int(timeout_seconds) if timeout_seconds else 300

        # Model selection
        model_used = ""
        if (model_mode or "auto") == "custom":
            model_used = (model_override or "").strip()
        if not model_used:
            model_used = _get_first_model_id(server_url, headers, timeout_s) or "local-model"

        # Stop selection
        stop_value = ""
        if stop_mode == "preset:common_eot":
            # Common EOT tokens seen across many templates
            stop_value = "<|eot_id|>"
        elif stop_mode == "preset:triple_hash":
            stop_value = "###"
        elif stop_mode == "custom":
            stop_value = (stop_custom or "").strip()
        # else none => ""

        endpoint = f"{server_url}/v1/chat/completions"

        messages = []
        sys = (system_prompt or "").strip()

        if json_mode:
            hint = (json_schema_hint or "").strip()
            json_instruction = "Return ONLY valid JSON. No commentary, no markdown, no code fences."
            if hint:
                json_instruction += f"\nJSON schema / shape hint:\n{hint}"
            sys = (sys + "\n\n" + json_instruction).strip() if sys else json_instruction

        if sys:
            messages.append({"role": "system", "content": sys})

        if image is not None:
            data_url = _image_tensor_to_base64_png(image)
            user_content = [{"type": "text", "text": user_text}]
            if data_url:
                user_content.append({"type": "image_url", "image_url": {"url": data_url}})
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_text})

        payload: Dict[str, Any] = {
            "model": model_used,
            "messages": messages,
        }

        if max_tokens and int(max_tokens) > 0:
            payload["max_tokens"] = int(max_tokens)
        if seed is not None and int(seed) >= 0:
            payload["seed"] = int(seed)
        if stop_value:
            payload["stop"] = stop_value

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        tick = _make_progress() if stream else None

        try:
            if stream:
                payload["stream"] = True
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s, stream=True)
                resp.encoding = "utf-8"
                if resp.status_code != 200:
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    raise RuntimeError(f"llama.cpp server error {resp.status_code}: {err}")

                content, reasoning, meta = _parse_stream_and_accumulate(resp, tick=tick)
                answer = _postprocess_text(_clean_answer(content), text_postprocess)
                thinking = _postprocess_text((reasoning or "").strip(), text_postprocess)

                raw = json.dumps({"stream_meta": meta, "content": content, "reasoning": reasoning}, ensure_ascii=False)
                raw = _postprocess_text(raw, text_postprocess)

                json_obj = _extract_json_from_text(answer) if json_mode else None
                json_str = json.dumps(json_obj, ensure_ascii=False, indent=2) if json_obj is not None else ""
                json_str = _postprocess_text(json_str, text_postprocess)

                return (answer, thinking, json_str, raw, model_used)

            else:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
                resp.encoding = "utf-8"
                if resp.status_code != 200:
                    try:
                        err = resp.json()
                    except Exception:
                        err = resp.text
                    raise RuntimeError(f"llama.cpp server error {resp.status_code}: {err}")

                data = resp.json()
                raw = json.dumps(data, ensure_ascii=False, indent=2)
                raw = _postprocess_text(raw, text_postprocess)

                choices = data.get("choices") or []
                if not choices:
                    return ("", "", "", raw, model_used)

                first = choices[0] if isinstance(choices[0], dict) else {}
                msg = first.get("message")

                if not isinstance(msg, dict):
                    text = first.get("text", "")
                    answer = _postprocess_text(_clean_answer(text), text_postprocess)
                    json_obj = _extract_json_from_text(answer) if json_mode else None
                    json_str = json.dumps(json_obj, ensure_ascii=False, indent=2) if json_obj is not None else ""
                    json_str = _postprocess_text(json_str, text_postprocess)
                    return (answer, "", json_str, raw, model_used)

                thinking, answer = _split_thinking_and_answer(msg)
                answer = _postprocess_text(answer, text_postprocess)
                thinking = _postprocess_text(thinking, text_postprocess)

                json_obj = _extract_json_from_text(answer) if json_mode else None
                json_str = json.dumps(json_obj, ensure_ascii=False, indent=2) if json_obj is not None else ""
                json_str = _postprocess_text(json_str, text_postprocess)

                return (answer, thinking, json_str, raw, model_used)

        except requests.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")


NODE_CLASS_MAPPINGS = {
    "SimpleLlamaCppClient": SimpleLlamaCppClient
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleLlamaCppClient": "Simple Llama.cpp Client (Chat + Vision + JSON + Stream)"
}
