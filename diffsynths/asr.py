"""
ASR helpers for the Gradio playground.

Backends are optional and lazy-loaded so the rest of the app can import without
pulling heavy deps.
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


class ASRProcessor:
    """Simple ASR processor wrapper with a unified API."""

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir
        self.current_model_id: Optional[str] = None
        self._backend: Any = None
        self._backend_kwargs: dict[str, Any] = {}

    def load_model(self, model_id: str, **kwargs: Any) -> None:
        if self.current_model_id == model_id and self._backend is not None:
            return

        self.unload_model()
        self._backend_kwargs = dict(kwargs)

        if model_id == "whisper":
            try:
                import whisper  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "whisper is not installed. Install with: `uv pip install openai-whisper`"
                ) from e

            whisper_model = kwargs.get("whisper_model", "base")
            self._backend = whisper.load_model(whisper_model)
        elif model_id in ("sensevoice", "paraformer"):
            raise NotImplementedError("FunASR backends are not wired yet in this repo.")
        else:
            raise ValueError(f"Unsupported ASR model: {model_id}")

        self.current_model_id = model_id
        logger.info("ASR model loaded: %s", model_id)

    def unload_model(self) -> None:
        self._backend = None
        self.current_model_id = None
        self._backend_kwargs = {}

    def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        if self._backend is None or self.current_model_id is None:
            raise RuntimeError("ASR model not loaded. Call load_model() first.")

        if self.current_model_id == "whisper":
            language = kwargs.get("language")
            result = self._backend.transcribe(audio_path, language=language)  # type: ignore[attr-defined]
            text = (result or {}).get("text", "")
            return str(text).strip()

        raise ValueError(f"Unsupported ASR model: {self.current_model_id}")


_processor: Optional[ASRProcessor] = None


def get_processor(model_dir: Optional[str] = None) -> ASRProcessor:
    global _processor
    if _processor is None:
        _processor = ASRProcessor(model_dir=model_dir)
    elif model_dir is not None:
        _processor.model_dir = model_dir
    return _processor


def unload_model() -> None:
    """Unload ASR model and release memory."""
    proc = get_processor()
    proc.unload_model()


def get_model_status() -> tuple[bool, Optional[str]]:
    """Return (loaded, current_model_id)."""
    proc = get_processor()
    return proc._backend is not None, proc.current_model_id


def transcribe_audio(
    audio_path: str,
    model_id: str,
    *,
    output_path: str,
    model_dir: Optional[str] = None,
    language: Optional[str] = None,
    whisper_model: str = "base",
) -> tuple[str, str]:
    """
    High-level helper for Gradio.

    Returns:
        (transcript, output_text_file_path)
    """
    proc = get_processor(model_dir=model_dir)
    if proc.current_model_id != model_id or proc._backend is None:
        proc.load_model(model_id, whisper_model=whisper_model)

    text = proc.transcribe(audio_path, language=language)
    out_file = _prepare_output_file(output_path, "asr", ".txt")
    Path(out_file).write_text(text, encoding="utf-8")
    return text, out_file
