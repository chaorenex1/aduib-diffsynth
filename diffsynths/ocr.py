"""
OCR helpers for the Gradio playground.

This module intentionally keeps dependencies optional and lazy-loaded.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _prepare_output_file(output_path: str, task_dir: str, ext: str) -> str:
    """
    Normalize output_path similar to diffsynths.text_to_image:
    - If output_path looks like a file (has suffix), treat it as the file path.
    - Otherwise treat it as a base directory and write to base/outputs/<task_dir>/uuid.ext
    """
    base = Path(output_path)
    if base.suffix:
        base.parent.mkdir(parents=True, exist_ok=True)
        return str(base)

    out_dir = base / "outputs" / task_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{uuid.uuid4()}{ext}")


class OCRProcessor:
    """Simple OCR processor wrapper with a unified API."""

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

        if model_id == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "paddleocr is not installed. Install with: `uv pip install -E ocr`"
                ) from e

            lang = kwargs.get("lang", "ch")
            use_angle_cls = bool(kwargs.get("use_angle_cls", True))
            self._backend = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, show_log=False)
        elif model_id == "paddleocr-cli":
            # Defer to CLI execution in process(); presence is checked there.
            self._backend = {"type": "cli"}
        elif model_id == "surya":
            # Surya OCR packages vary; keep as a placeholder until wired.
            raise NotImplementedError("surya OCR backend is not wired yet in this repo.")
        else:
            raise ValueError(f"Unsupported OCR model: {model_id}")

        self.current_model_id = model_id
        logger.info("OCR model loaded: %s", model_id)

    def unload_model(self) -> None:
        self._backend = None
        self.current_model_id = None
        self._backend_kwargs = {}

    def process(self, image_path: str, **kwargs: Any) -> str:
        if self._backend is None or self.current_model_id is None:
            raise RuntimeError("OCR model not loaded. Call load_model() first.")

        model_id = self.current_model_id
        if model_id == "paddleocr":
            # PaddleOCR returns list[list[...]]; flatten and join by newline.
            result = self._backend.ocr(image_path, cls=True)  # type: ignore[attr-defined]
            lines: list[str] = []
            for page in result or []:
                for item in page or []:
                    if not item or len(item) < 2:
                        continue
                    text_score = item[1]
                    if isinstance(text_score, (list, tuple)) and text_score:
                        text = str(text_score[0])
                        if text.strip():
                            lines.append(text.strip())
            return "\n".join(lines).strip()

        if model_id == "paddleocr-cli":
            import json
            import subprocess

            cmd = ["paddleocr", "--image_dir", image_path, "--use_angle_cls", "true", "--show_log", "false", "--output", "json"]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            except FileNotFoundError as e:
                raise RuntimeError("paddleocr CLI not found on PATH.") from e

            # Best-effort parse: expect a JSON object or list in stdout.
            try:
                payload = json.loads(proc.stdout)
            except Exception:
                return proc.stdout.strip()

            # Common payload shapes differ; attempt a small set.
            if isinstance(payload, dict) and "text" in payload:
                return str(payload["text"]).strip()
            if isinstance(payload, list):
                return "\n".join(str(x) for x in payload).strip()
            return str(payload).strip()

        raise ValueError(f"Unsupported OCR model: {model_id}")


_processor: Optional[OCRProcessor] = None


def get_processor(model_dir: Optional[str] = None) -> OCRProcessor:
    global _processor
    if _processor is None:
        _processor = OCRProcessor(model_dir=model_dir)
    elif model_dir is not None:
        # Allow late binding from the app (e.g. gradio_app.get_diffsynth_model_dir()).
        _processor.model_dir = model_dir
    return _processor


def unload_model() -> None:
    """Unload OCR model and release memory."""
    proc = get_processor()
    proc.unload_model()


def get_model_status() -> tuple[bool, Optional[str]]:
    """Return (loaded, current_model_id)."""
    proc = get_processor()
    return proc._backend is not None, proc.current_model_id


def process_image(
    image_path: str,
    model_id: str,
    *,
    output_path: str,
    model_dir: Optional[str] = None,
    lang: str = "ch",
) -> tuple[str, str]:
    """
    High-level helper for Gradio.

    Returns:
        (recognized_text, output_text_file_path)
    """
    proc = get_processor(model_dir=model_dir)
    if proc.current_model_id != model_id or proc._backend is None:
        proc.load_model(model_id, lang=lang)

    text = proc.process(image_path)
    out_file = _prepare_output_file(output_path, "ocr", ".txt")
    Path(out_file).write_text(text, encoding="utf-8")
    return text, out_file
