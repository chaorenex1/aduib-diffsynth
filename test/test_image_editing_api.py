"""
图片编辑API测试示例

演示如何使用图片编辑接口
"""

import requests
import os

# API基础URL
BASE_URL = "http://localhost:8000"


def test_edit_by_upload():
    """测试上传图片编辑接口"""
    print("=" * 60)
    print("测试1: 上传图片编辑")
    print("=" * 60)

    url = f"{BASE_URL}/text-to-image/edit"

    # 准备测试图片
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        print("请准备一张测试图片并命名为 test_image.jpg")
        return

    # 准备请求
    files = {
        'input_image': open(test_image_path, 'rb')
    }

    data = {
        'prompt': '把天空变成日落的颜色，增加温暖的光线',
        'negative_prompt': '模糊,低质量,变形',
        'model_type': 'Qwen-Image-Edit-2509',
        'width': 1024,
        'height': 1024,
        'num_inference_steps': 30,  # 使用较少步数加快测试
        'guidance_scale': 4.5,
        'offload_model': False
    }

    try:
        print(f"📤 发送请求到: {url}")
        print(f"📝 提示词: {data['prompt']}")

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 编辑成功!")
            print(f"📁 输出图片路径: {result['image_path']}")
            print(f"💬 消息: {result['message']}")
            return result['image_path']
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

    finally:
        files['input_image'].close()


def test_edit_by_path(image_path):
    """测试通过路径编辑图片接口"""
    print("\n" + "=" * 60)
    print("测试2: 通过路径编辑图片")
    print("=" * 60)

    url = f"{BASE_URL}/text-to-image/edit-by-path"

    if not image_path or not os.path.exists(image_path):
        print(f"❌ 图片路径不存在: {image_path}")
        return

    payload = {
        "input_image_path": image_path,
        "prompt": "添加一些梦幻的光斑效果",
        "negative_prompt": "模糊,低质量",
        "model_type": "Qwen-Image-Edit-2509",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 30,
        "guidance_scale": 4.5,
        "offload_model": False
    }

    try:
        print(f"📤 发送请求到: {url}")
        print(f"📝 提示词: {payload['prompt']}")
        print(f"📁 输入图片: {image_path}")

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 编辑成功!")
            print(f"📁 输出图片路径: {result['image_path']}")
            print(f"💬 消息: {result['message']}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")


def test_load_model():
    """测试加载模型接口"""
    print("\n" + "=" * 60)
    print("测试3: 预加载模型")
    print("=" * 60)

    url = f"{BASE_URL}/text-to-image/load-model"

    payload = {
        "model_type": "Qwen-Image-Edit-2509",
        "lora_model": "",
        "offload_model": False
    }

    try:
        print(f"📤 发送请求到: {url}")
        print(f"🔧 模型类型: {payload['model_type']}")

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 模型加载成功!")
            print(f"💬 消息: {result['message']}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")


def test_unload_model():
    """测试卸载模型接口"""
    print("\n" + "=" * 60)
    print("测试4: 卸载模型")
    print("=" * 60)

    url = f"{BASE_URL}/text-to-image/unload-model"

    try:
        print(f"📤 发送请求到: {url}")

        response = requests.post(url)

        if response.status_code == 200:
            result = response.json()
            print(f"✅ 模型卸载成功!")
            print(f"💬 消息: {result['message']}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")


def main():
    """主测试函数"""
    print("\n🚀 图片编辑API测试开始\n")

    # 检查服务器是否运行
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print("✅ API服务器正在运行")
    except Exception as e:
        print(f"❌ 无法连接到API服务器: {e}")
        print("请确保服务器正在运行: python app.py")
        return

    # 测试1: 上传图片编辑
    output_path = test_edit_by_upload()

    # 测试2: 通过路径编辑（使用上一步的输出）
    if output_path:
        test_edit_by_path(output_path)

    # 测试3: 预加载模型
    # test_load_model()

    # 测试4: 卸载模型
    # test_unload_model()

    print("\n" + "=" * 60)
    print("✨ 测试完成!")
    print("=" * 60)
    print("\n💡 提示:")
    print("  - 首次运行会下载模型，可能需要较长时间")
    print("  - 可以通过修改 num_inference_steps 来平衡速度和质量")
    print("  - 查看完整API文档: docs/IMAGE_EDITING_API.md")
    print("  - API交互式文档: http://localhost:8000/docs")


if __name__ == "__main__":
    main()

