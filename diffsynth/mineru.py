import logging
import os
import shutil
import uuid
from html import escape
from pathlib import Path

import torch
from mineru.cli.common import read_fn, do_parse, pdf_suffixes, image_suffixes
from mineru.utils.config_reader import get_device

from utils.markdown import process_markdown_file

logger= logging.getLogger(__name__)


def parse_pdf(input_path, output_dir, method,lang):
    """Parse PDF documents in the specified directory."""
    backend = os.getenv('MINERU_BACKEND', 'vlm-transformers')
    device_mode = os.getenv('MINERU_DEVICE_MODE', 'cuda')
    model_source = os.getenv('MINERU_MODEL_SOURCE', 'modelscope')
    if not backend.endswith('-client'):
        def get_device_mode() -> str:
            if device_mode is not None:
                return device_mode
            else:
                return get_device()
        if os.getenv('MINERU_DEVICE_MODE', None) is None:
            os.environ['MINERU_DEVICE_MODE'] = get_device_mode()

        if os.getenv('MINERU_MODEL_SOURCE', None) is None:
            os.environ['MINERU_MODEL_SOURCE'] = model_source

    os.makedirs(output_dir, exist_ok=True)

    def parse_doc(path_list: list[Path]):
        try:
            file_name_list = []
            pdf_bytes_list = []
            lang_list = []
            for path in path_list:
                file_name = str(Path(path).stem)
                pdf_bytes = read_fn(path)
                file_name_list.append(file_name)
                pdf_bytes_list.append(pdf_bytes)
                lang_list.append(lang)

            do_parse(
                output_dir=output_dir,
                pdf_file_names=file_name_list,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=backend,
                parse_method=method,
                start_page_id=0,
                end_page_id=None
            )
        except Exception as e:
            logger.exception(e)

    if os.path.isdir(input_path):
        doc_path_list = []
        for doc_path in Path(input_path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                doc_path_list.append(doc_path)
        parse_doc(doc_path_list)
    else:
        parse_doc([Path(input_path)])

    torch.cuda.empty_cache()

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

    def resolve_path(file_obj):
        if isinstance(file_obj, dict):
            return file_obj.get("name") or file_obj.get("path")
        return getattr(file_obj, "name", file_obj)

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
        new_md_file=md_file.parent / f"{md_file.stem}_uploaded.md"
        from component.storage.base_storage import storage_manager
        result = process_markdown_file(str(md_file), str(new_md_file), storage_manager.storage_instance, "blog_static")
        if result:
            logger.debug(f"Markdown file processed result: {result}")
            return f"<div>Markdown file '{md_file.name}' uploaded successfully.</div>"

    return "<div>No markdown file found to upload.</div>"
