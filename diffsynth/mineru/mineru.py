import logging,os
from pathlib import Path

from mineru.cli.common import read_fn, do_parse, pdf_suffixes, image_suffixes
from mineru.utils.config_reader import get_device

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
