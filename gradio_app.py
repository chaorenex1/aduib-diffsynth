"""Minimal Gradio UI to demo DiffSynth functionality."""

import gradio as gr

from app_factory import create_app
from diffsynth.diffsynth import process_pdf_files, upload_to_blog, create_blog

app=create_app()
mineru_working_dir = app.app_home + "/mineru"


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
                    # 5. 下载 ZIP 文件
                    download_file = gr.File(label="Download Results (ZIP)")
                    # 6. 上传blog
                    upload_file_button = gr.Button("Upload to Blog")
                    upload_file_button.click(
                        fn=upload_to_blog,
                        inputs=[download_file],
                        outputs=[output_box]
                    )
            submit_button.click(
                fn=process_pdf_files,
                inputs=[files, lang, method],
                outputs=[output_box, download_file],
            )
        with gr.Tab("Aduib Blog RAG"):
            with gr.Column(scale=1):
                # 1. 选择 Markdown 文件
                md_files = gr.Files(label="Upload Markdown File", file_types=[".md"])
                upload_md_button = gr.Button("Upload Markdown to Blog")
                md_output_box = gr.HTML(label="Output", value="<div>等待上传文件...</div>")
            upload_md_button.click(
                fn=create_blog,
                inputs=[md_files],
                outputs=[md_output_box],
            )
    return gradio_app


def main():
    gradio_app = build_interface()
    gradio_app.launch(server_name="0.0.0.0", server_port=7860, show_error=True,mcp_server=True,allowed_paths=[app.app_home])


if __name__ == "__main__":
    main()