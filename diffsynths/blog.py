import logging
import os
import shutil
import uuid
from html import escape
from pathlib import Path

from diffsynths.aduib_ai import create_paragraph_rag
from diffsynths.mineru import parse_pdf
from utils.markdown import process_markdown_file

logger= logging.getLogger(__name__)


def resolve_path(file_obj):
    if isinstance(file_obj, dict):
        return file_obj.get("name") or file_obj.get("path")
    return getattr(file_obj, "name", file_obj)

def process_pdf_files(files, lang, method):
    from gradio_app import mineru_working_dir

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

    if not files:
        yield (build_progress_html(0, 1, "No files uploaded."), None)
        return

    output_id = str(uuid.uuid4())
    output_dir = os.path.join(mineru_working_dir, output_id)
    total_files = len(files)
    yield (build_progress_html(0, total_files, "Starting PDF parsing..."), None)

    for idx, file_obj in enumerate(files, start=1):
        file_path = resolve_path(file_obj)
        if not file_path:
            continue
        file_name = os.path.basename(file_path)
        yield (build_progress_html(idx - 1, total_files, f"Processing {file_name} ({idx}/{total_files})"),None)
        parse_pdf(
            input_path=file_path,
            output_dir=output_dir,
            method=method,
            lang=lang,
        )
        yield (build_progress_html(idx, total_files, f"Finished {file_name} ({idx}/{total_files})"),None)

    result_files = os.listdir(output_dir)
    result_text = (
        "Processed files:\n" + "\n".join(result_files)
        if result_files
        else "Processing complete but no output files were generated."
    )
    # 打包为 zip，供下载
    zip_path = shutil.make_archive(output_dir, "zip", root_dir=output_dir)

    # 返回最终的 HTML 和 zip 文件路径（gr.File 会作为可下载文件显示）
    yield (build_progress_html(total_files, total_files, "All tasks complete.", result_text), zip_path)


def upload_to_blog(zip_file):
    if not zip_file:
        return "<div>No file to upload.</div>"

    from gradio_app import mineru_working_dir
    #1.获取output_id
    zip_path = zip_file.name
    output_id = Path(zip_path).stem  # 去掉 .zip 后缀
    output_dir = os.path.join(mineru_working_dir, output_id)
    #2. 深度遍历获取md文件
    md_files = list(Path(output_dir).rglob("*.md"))
    if not md_files:
        return "<div>No markdown file found in the output.</div>"
    for md_file in md_files:
        # 3. 读取md内容
        new_md_file = md_file.parent / f"{md_file.stem}_uploaded.md"
        new_md_file_name = os.path.basename(new_md_file)
        from component.storage.base_storage import storage_manager
        result = process_markdown_file(str(md_file), str(new_md_file), storage_manager.storage_instance, "blog_static")
        if result:
            with open(new_md_file, "rb") as f:
                md_content = f.read()
            # 4. 创建RAG
            create_paragraph_rag(
                file_content=md_content,
                file_name=new_md_file_name,
            )
            logger.debug(f"Markdown file processed result: {result}")
    return f"<div>Markdown file '{len(md_files)}' uploaded successfully.</div>"

def create_blog(files):
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

    if not files:
        yield build_progress_html(0, 1, "No markdown file uploaded.")
        return

    total_files = len(files)
    yield build_progress_html(0, total_files, "Starting to upload markdown files to blog...")

    uploaded_files = []
    for idx, file_obj in enumerate(files, start=1):
        file_path = resolve_path(file_obj)
        file_name = os.path.basename(file_path)

        yield build_progress_html(idx - 1, total_files, f"Processing {file_name} ({idx}/{total_files})")

        with open(file_path, "rb") as f:
            md_content = f.read()
        # 创建RAG
        create_paragraph_rag(
            file_content=md_content,
            file_name=file_name,
        )
        logger.debug(f"Markdown file '{file_name}' uploaded to blog.")
        uploaded_files.append(file_name)

        yield build_progress_html(idx, total_files, f"Finished {file_name} ({idx}/{total_files})")

    result_text = "Uploaded files:\n" + "\n".join(uploaded_files)
    yield build_progress_html(total_files, total_files, "All markdown files uploaded successfully.", result_text)
