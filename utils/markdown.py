import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


def extract_titles(markdown_content: str) -> List[Dict[str, str]]:
    """
    从Markdown内容中提取所有标题

    Args:
        markdown_content: Markdown文本内容

    Returns:
        包含标题信息的字典列表，每个字典包含:
        - level: 标题级别 (1-6)
        - text: 标题文本
        - raw: 原始标题行
    """
    titles = []
    # 匹配标题的正则表达式 (支持 # 标题格式)
    pattern = r'^(#{1,6})\s+(.+)$'

    for line in markdown_content.split('\n'):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            level = len(match.group(1))
            text = match.group(2).strip()
            titles.append({
                'level': level,
                'text': text,
                'raw': line
            })

    return titles


def extract_images(markdown_content: str) -> List[Dict[str, str]]:
    """
    从Markdown内容中提取所有图片信息

    Args:
        markdown_content: Markdown文本内容

    Returns:
        包含图片信息的字典列表，每个字典包含:
        - alt: 图片的alt文本
        - url: 图片的URL或路径
        - raw: 原始的markdown图片语法
    """
    images = []
    # 匹配图片的正则表达式: ![alt](url)
    pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'

    for match in re.finditer(pattern, markdown_content):
        images.append({
            'alt': match.group(1),
            'url': match.group(2),
            'raw': match.group(0)
        })

    return images


def is_local_path(image_url: str) -> bool:
    """
    判断图片URL是否为本地路径

    Args:
        image_url: 图片URL或路径

    Returns:
        True if local path, False if URL
    """
    # 检查是否为URL
    parsed = urlparse(image_url)
    if parsed.scheme in ('http', 'https', 'ftp'):
        return False

    # 检查是否为本地路径
    return True


def read_image_data(image_path: str, base_dir: Optional[str] = None) -> Optional[bytes]:
    """
    读取本地图片数据

    Args:
        image_path: 图片路径
        base_dir: 基础目录，用于解析相对路径

    Returns:
        图片的二进制数据，如果读取失败则返回None
    """
    try:
        if base_dir and not os.path.isabs(image_path):
            image_path = os.path.join(base_dir, image_path)

        with open(image_path, 'rb') as f:
            return f.read()
    except Exception as e:
        log.error(f"Failed to read image from {image_path}: {e}")
        return None


def download_image(image_url: str) -> Optional[bytes]:
    """
    从URL下载图片

    Args:
        image_url: 图片URL

    Returns:
        图片的二进制数据，如果下载失败则返回None
    """
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        log.error(f"Failed to download image from {image_url}: {e}")
        return None


def get_image_extension(image_url: str) -> str:
    """
    从图片URL或路径获取文件扩展名

    Args:
        image_url: 图片URL或路径

    Returns:
        文件扩展名（如 .jpg, .png）
    """
    path = urlparse(image_url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else '.jpg'


def replace_images_with_urls(
    markdown_content: str,
    storage_instance,
    base_dir: Optional[str] = None,
    storage_prefix: str = 'markdown_images',
    filename_generator: Optional[callable] = None
) -> Tuple[str, List[Dict]]:
    """
    解析Markdown中的图片，上传到图床并替换链接

    Args:
        markdown_content: Markdown文本内容
        storage_instance: 存储实例（BaseStorage的实现）
        base_dir: 基础目录，用于解析相对路径
        storage_prefix: 存储路径前缀
        filename_generator: 文件名生成器函数，接收原始URL返回新文件名
                           如果为None，则使用UUID生成

    Returns:
        元组:
        - 替换后的Markdown内容
        - 上传结果列表，每项包含:
          - original_url: 原始URL
          - new_url: 新URL
          - success: 是否成功
          - error: 错误信息（如果失败）
    """
    from .uuid import random_uuid

    images = extract_images(markdown_content)
    upload_results = []
    new_content = markdown_content

    for image in images:
        result = {
            'original_url': image['url'],
            'new_url': None,
            'success': False,
            'error': None
        }

        try:
            # 获取图片数据
            image_data = None

            if is_local_path(image['url']):
                # 本地路径
                image_data = read_image_data(image['url'], base_dir)
            else:
                # 远程URL - 已经是URL，可以选择下载后重新上传或直接保留
                # 这里选择下载后上传，确保图片在自己的存储中
                image_data = download_image(image['url'])

            if not image_data:
                result['error'] = 'Failed to read/download image'
                upload_results.append(result)
                continue

            # 生成存储文件名
            if filename_generator:
                filename = filename_generator(image['url'])
            else:
                ext = get_image_extension(image['url'])
                filename = f"{random_uuid()}{ext}"

            # 构建完整的存储路径
            storage_path = f"{storage_prefix}/{filename}"

            # 上传到存储
            storage_instance.save(storage_path, image_data)

            # 生成新的URL（这里假设存储服务提供公共访问URL）
            # 需要根据实际的存储配置来构建URL
            # 如果是S3，可以构建S3的公共URL
            new_url = construct_public_url(storage_instance, storage_path)

            result['new_url'] = new_url
            result['success'] = True

            # 替换Markdown中的图片链接
            new_image_syntax = f"![{image['alt']}]({new_url})"
            new_content = new_content.replace(image['raw'], new_image_syntax)

            log.info(f"Successfully uploaded image: {image['url']} -> {new_url}")

        except Exception as e:
            result['error'] = str(e)
            log.error(f"Failed to process image {image['url']}: {e}")

        upload_results.append(result)

    return new_content, upload_results


def construct_public_url(storage_instance, storage_path: str) -> str:
    """
    根据存储实例和路径构建公共访问URL

    Args:
        storage_instance: 存储实例
        storage_path: 存储路径

    Returns:
        公共访问URL
    """
    # 根据不同的存储类型构建URL
    from configs import config

    # 如果是S3存储
    if hasattr(storage_instance, 'bucket_name'):
        # S3风格的URL
        bucket_name = storage_instance.bucket_name
        endpoint = config.S3_ENDPOINT

        # 移除endpoint末尾的斜杠
        endpoint = endpoint.rstrip('/')

        # 构建URL
        return f"{endpoint}/{bucket_name}/{storage_path}"

    # 默认返回路径
    return f"/{storage_path}"


def process_markdown_file(
    input_file: str,
    output_file: str,
    storage_instance,
    storage_prefix: str = 'markdown_images'
) -> Dict:
    """
    处理Markdown文件：读取、上传图片、替换链接并保存

    Args:
        input_file: 输入Markdown文件路径
        output_file: 输出Markdown文件路径
        storage_instance: 存储实例
        storage_prefix: 存储路径前缀

    Returns:
        处理结果字典:
        - success: 是否成功
        - titles: 提取的标题列表
        - upload_results: 图片上传结果列表
        - error: 错误信息（如果失败）
    """
    result = {
        'success': False,
        'titles': [],
        'upload_results': [],
        'error': None
    }

    try:
        # 读取Markdown文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取标题
        result['titles'] = extract_titles(content)

        # 获取文件所在目录（用于解析相对路径）
        base_dir = os.path.dirname(os.path.abspath(input_file))

        # 处理图片
        new_content, upload_results = replace_images_with_urls(
            content,
            storage_instance,
            base_dir=base_dir,
            storage_prefix=storage_prefix
        )

        result['upload_results'] = upload_results

        # 保存处理后的Markdown文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        result['success'] = True
        log.info(f"Successfully processed markdown file: {input_file} -> {output_file}")

    except Exception as e:
        result['error'] = str(e)
        log.error(f"Failed to process markdown file {input_file}: {e}")

    return result

