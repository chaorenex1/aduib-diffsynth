"""
示例：如何使用 markdown.py 中的功能

演示如何：
1. 从Markdown中提取标题
2. 解析图片并上传到图床
3. 替换图片链接
"""
from utils.markdown import (
    extract_titles,
    extract_images,
    replace_images_with_urls,
    process_markdown_file
)
from component.storage.s3_storage import S3Storage


def example_extract_titles():
    """示例：提取Markdown标题"""
    markdown_content = """
# 主标题
这是一些内容
## 二级标题
更多内容
### 三级标题
    """

    titles = extract_titles(markdown_content)
    print("提取的标题：")
    for title in titles:
        indent = "  " * (title['level'] - 1)
        print(f"{indent}- {title['text']} (级别{title['level']})")


def example_extract_images():
    """示例：提取Markdown中的图片"""
    markdown_content = """
# 文档标题
这是一张图片：
![示例图片](./images/example.png)

另一张网络图片：
![网络图片](https://example.com/image.jpg)
    """

    images = extract_images(markdown_content)
    print("\n提取的图片：")
    for img in images:
        print(f"- Alt: {img['alt']}")
        print(f"  URL: {img['url']}")


def example_replace_images():
    """示例：上传图片并替换链接"""
    markdown_content = """
# 测试文档
![本地图片](./test.png)
![远程图片](https://example.com/image.jpg)
    """

    # 初始化存储实例（需要配置好S3或其他存储）
    # storage = S3Storage()

    # 注意：这个示例需要实际的存储配置才能运行
    # new_content, results = replace_images_with_urls(
    #     markdown_content,
    #     storage,
    #     base_dir='./test_files',
    #     storage_prefix='markdown_images'
    # )

    # print("\n替换后的内容：")
    # print(new_content)
    # print("\n上传结果：")
    # for result in results:
    #     print(f"原始: {result['original_url']}")
    #     print(f"新的: {result['new_url']}")
    #     print(f"成功: {result['success']}")
    #     if result['error']:
    #         print(f"错误: {result['error']}")

    print("\n注意：需要配置存储实例才能实际运行图片上传功能")


def example_process_file():
    """示例：完整处理Markdown文件"""
    # 注意：这个示例需要实际的存储配置和文件才能运行
    # storage = S3Storage()

    # result = process_markdown_file(
    #     input_file='./input.md',
    #     output_file='./output.md',
    #     storage_instance=storage,
    #     storage_prefix='markdown_images'
    # )

    # if result['success']:
    #     print("\n处理成功！")
    #     print(f"提取的标题数量: {len(result['titles'])}")
    #     print(f"处理的图片数量: {len(result['upload_results'])}")
    # else:
    #     print(f"处理失败: {result['error']}")

    print("\n注意：需要配置存储实例和准备测试文件才能实际运行")


if __name__ == '__main__':
    print("=== Markdown 工具示例 ===\n")

    # 运行示例
    example_extract_titles()
    example_extract_images()
    example_replace_images()
    example_process_file()

    print("\n=== 使用说明 ===")
    print("1. extract_titles(content) - 提取所有标题")
    print("2. extract_images(content) - 提取所有图片")
    print("3. replace_images_with_urls(content, storage, ...) - 上传图片并替换链接")
    print("4. process_markdown_file(input, output, storage) - 完整处理文件")

