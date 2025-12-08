"""Minimal Gradio UI to demo DiffSynth functionality."""
from __future__ import annotations
import gradio as gr
from html import escape

from diffsynth.mineru.mineru import parse_pdf


def process_files(files, lang, method):
    import tempfile
    import os

    def build_progress_html(current: int, total: int, status: str, body: str | None = None) -> str:
        total = max(total, 1)
        body_section = f"<pre>{escape(body)}</pre>" if body else ""
        return (
            "<div style='display:flex;flex-direction:column;gap:0.5rem;'>"
            f"<progress value='{current}' max='{total}' style='width:100%;'></progress>"
            f"<div>{escape(status)}</div>"
            f"{body_section}"
            "</div>"
        )

    def resolve_path(file_obj):
        if isinstance(file_obj, dict):
            return file_obj.get("name") or file_obj.get("path")
        return getattr(file_obj, "name", file_obj)

    if not files:
        yield build_progress_html(0, 1, "No files uploaded.")
        return

    output_dir = tempfile.mkdtemp()
    total_files = len(files)
    yield build_progress_html(0, total_files, "Starting PDF parsing...")

    for idx, file_obj in enumerate(files, start=1):
        file_path = resolve_path(file_obj)
        if not file_path:
            continue
        file_name = os.path.basename(file_path)
        yield build_progress_html(idx - 1, total_files, f"Processing {file_name} ({idx}/{total_files})")
        parse_pdf(
            input_path=file_path,
            output_dir=output_dir,
            method=method,
            lang=lang,
        )
        yield build_progress_html(idx, total_files, f"Finished {file_name} ({idx}/{total_files})")

    result_files = os.listdir(output_dir)
    result_text = (
        "Processed files:\n" + "\n".join(result_files)
        if result_files
        else "Processing complete but no output files were generated."
    )
    yield build_progress_html(total_files, total_files, "All tasks complete.", result_text)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="DiffSynth Playground") as gradio_app:
        gr.Markdown("# DiffSynth Playground\n")
        # mineru tab页
        with gr.Tab("MINERU"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 1. 选择文件或者目录
                    files = gr.Files(label="Upload PDF Files", file_types=[".pdf"])
                    # 2. 选择语言
                    lang = gr.Dropdown(
                        choices=["auto", "en", "zh"],
                        value="zh",
                        label="Select Language",
                    )
                    # 3. 选择方式
                    method = gr.Dropdown(
                        choices=["auto", "txt", "ocr"],
                        value="ocr",
                        label="Select Parsing Method",
                    )
                    submit_button = gr.Button("Submit")
                with gr.Column(scale=1):
                    # 4. 显示结果
                    output_box = gr.HTML(label="Output", value="<div>等待上传文件...</div>")
            submit_button.click(
                fn=process_files,
                inputs=[files, lang, method],
                outputs=output_box,
            )
    return gradio_app


def main():
    gradio_app = build_interface()
    gradio_app.launch(server_name="0.0.0.0", server_port=7860, show_error=True,mcp_server=True)


if __name__ == "__main__":
    main()
