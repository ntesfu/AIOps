"""Pluggable VLM backends — the primary swap point.

A backend turns a ``Prompt`` into raw text. Swap freely: ``MockVLMBackend`` for
offline development/tests, ``ClaudeVLMBackend`` for the Anthropic API,
``QwenVLBackend`` for a local Qwen-VL on the GPU box. The pipeline depends only on
the ``VLMBackend`` protocol, never on a concrete model.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from aiops.adjudication.types import Prompt


@runtime_checkable
class VLMBackend(Protocol):
    name: str

    def generate(self, prompt: Prompt) -> str:
        """Return the model's raw text response for ``prompt``."""
        ...


class MockVLMBackend:
    """Deterministic offline backend. Echoes the detector prior into a schema-valid
    JSON response (optionally fenced) so the whole pipeline runs with no API/GPU."""

    name = "mock"

    def __init__(self, *, fenced: bool = True, force_family: str | None = None) -> None:
        self.fenced = fenced
        self.force_family = force_family

    def generate(self, prompt: Prompt) -> str:
        pkt = prompt.packet
        cand = pkt.candidate if pkt else None
        family = self.force_family or (cand.detector_category if cand else None) or "Other"
        score = cand.detector_score if cand else 0.5
        step = cand.step_description if cand else ""
        payload = {
            "has_error": True,
            "mistake_family": family,
            "mistake_subtype": "",
            "description": f"Likely {family.lower()} while performing: {step}",
            "expected": pkt.expected_action if pkt and pkt.expected_action else step,
            "observed": "deviation from the expected step",
            "violated_role": "",
            "evidence": ", ".join(f.role for f in (pkt.frames if pkt else [])) or "n/a",
            "confidence": round(float(max(0.0, min(1.0, score))), 3),
            "rationale": "mock backend: echoing detector prior",
        }
        body = json.dumps(payload)
        return f"```json\n{body}\n```" if self.fenced else body


class ClaudeVLMBackend:
    """Anthropic Messages API backend (lazy import). Fills in later; the structure
    is complete so it drops into the pipeline unchanged."""

    name = "claude"

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1024,
                 client: object | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - optional dep
                raise RuntimeError("pip install anthropic to use ClaudeVLMBackend") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def _image_blocks(self, prompt: Prompt) -> list[dict]:
        # Resolve FrameRefs to base64 image blocks. Left as a hook: an evidence
        # builder that produced data-URI/base64 refs plugs in here.
        blocks = []
        for fr in prompt.images:
            if fr.ref.startswith("data:") or fr.ref.startswith("base64:"):
                b64 = fr.ref.split(",", 1)[-1].replace("base64:", "")
                blocks.append({"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}})
        return blocks

    def generate(self, prompt: Prompt) -> str:  # pragma: no cover - needs network
        client = self._get_client()
        content = [*self._image_blocks(prompt), {"type": "text", "text": prompt.user}]
        msg = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=prompt.system, messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class QwenVLBackend:
    """Local Qwen2.5-VL backend on the GPU box (lazy import)."""

    name = "qwen-vl"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 device: str | None = None, max_new_tokens: int = 512) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None

    def _load(self):  # pragma: no cover - GPU only
        if self._model is None:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.bfloat16,
                device_map=self.device or "auto").eval()

    def generate(self, prompt: Prompt) -> str:  # pragma: no cover - GPU only
        self._load()
        images = [fr.ref for fr in prompt.images]
        messages = [{"role": "system", "content": prompt.system},
                    {"role": "user", "content": [{"type": "text", "text": prompt.user},
                                                 *[{"type": "image", "image": im} for im in images]]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=images or None, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = out[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(gen, skip_special_tokens=True)[0]
