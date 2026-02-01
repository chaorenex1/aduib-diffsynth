"""
TTS helpers for the Gradio playground.

Backends are optional and lazy-loaded. The simplest supported backend here is
`edge-tts` (pip package imports as `edge_tts`).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _prepare_output_file(output_path: str, task_dir: str, ext: str) -> str:
    base = Path(output_path)
    if base.suffix:
        base.parent.mkdir(parents=True, exist_ok=True)
        return str(base)

    out_dir = base / "outputs" / task_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{uuid.uuid4()}{ext}")


class TTSGenerator:
    """Simple TTS generator wrapper with a unified API."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self.current_model_id: Optional[str] = None
        self._backend: Any = None
        self._backend_kwargs: dict[str, Any] = {}

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        # Many TTS backends are stateless; keep a consistent API anyway.
        if self.current_model_id == model_id:
            return

        self.unload_model()
        self._backend_kwargs = dict(kwargs)

        if model_id == "edge-tts":
            try:
                import edge_tts  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "edge-tts is not installed. Install with: `uv pip install edge-tts`"
                ) from e
            self._backend = edge_tts
        elif model_id in ("chattts", "cosyvoice", "fishspeech"):
            raise NotImplementedError(f"{model_id} backend is not wired yet in this repo.")
        else:
            raise ValueError(f"Unsupported TTS model: {model_id}")

        self.current_model_id = model_id
        logger.info("TTS model loaded: %s", model_id)

    def unload_model(self) -> None:
        self._backend = None
        self.current_model_id = None
        self._backend_kwargs = {}

    def generate(self, text: str, output_file: str, **kwargs: Any) -> str:
        if self._backend is None or self.current_model_id is None:
            raise RuntimeError("TTS model not loaded. Call load_model() first.")

        if self.current_model_id == "edge-tts":
            import asyncio

            voice = kwargs.get("voice", "zh-CN-XiaoxiaoNeural")
            rate = kwargs.get("rate", "+0%")
            volume = kwargs.get("volume", "+0%")

            async def _run() -> None:
                communicate = self._backend.Communicate(text, voice=voice, rate=rate, volume=volume)  # type: ignore[attr-defined]
                await communicate.save(output_file)

            asyncio.run(_run())
            return output_file

        raise ValueError(f"Unsupported TTS model: {self.current_model_id}")


_generator: Optional[TTSGenerator] = None


def get_generator(model_dir: Optional[str] = None) -> TTSGenerator:
    global _generator
    if _generator is None:
        _generator = TTSGenerator(model_dir=model_dir)
    elif model_dir is not None:
        _generator.model_dir = model_dir
    return _generator


def unload_model() -> None:
    """Unload TTS model and release memory."""
    gen = get_generator()
    gen.unload_model()


def get_model_status() -> tuple[bool, Optional[str]]:
    """Return (loaded, current_model_id)."""
    gen = get_generator()
    return gen._backend is not None, gen.current_model_id


def generate_speech(
    text: str,
    model_id: str,
    *,
    output_path: str,
    model_dir: Optional[str] = None,
    voice: str = "zh-CN-XiaoxiaoNeural",
    rate: str = "+0%",
    volume: str = "+0%",
) -> str:
    """
    High-level helper for Gradio.

    Returns:
        audio_file_path
    """
    gen = get_generator(model_dir=model_dir)
    if gen.current_model_id != model_id or gen._backend is None:
        gen.load_model(model_id)

    out_file = _prepare_output_file(output_path, "tts", ".mp3")
    gen.generate(text, out_file, voice=voice, rate=rate, volume=volume)
    return out_file
