"""
文生图功能测试脚本

使用示例：
python test_text_to_image.py
"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_generation():
    """测试简单的图像生成"""
    print("\n=== 测试 1: 简单图像生成 ===")

    try:
        from diffsynths.text_to_image import generate_image

        print("正在生成图像...")
        image_path = generate_image(
            prompt="a beautiful sunset over mountains, highly detailed, 4k",
            negative_prompt="low quality, blurry, deformed",
            model_type="sd",
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42,
        )

        print(f"✅ 图像生成成功！")
        print(f"   保存路径: {image_path}")
        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("   请安装 diffsynth-engine: uv pip install diffsynth-engine")
        return False
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return False


def test_batch_generation():
    """测试批量生成"""
    print("\n=== 测试 2: 批量图像生成 ===")

    try:
        from diffsynths.text_to_image import get_generator

        generator = get_generator()
        print("正在加载模型...")
        generator.load_model("sd")

        prompts = [
            "a red apple on a table",
            "a blue car in the street",
            "a green tree in the park",
        ]

        print(f"正在批量生成 {len(prompts)} 张图像...")
        image_paths = generator.batch_generate(
            prompts=prompts,
            width=512,
            height=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            seed=42,
        )

        print(f"✅ 批量生成成功！")
        for i, path in enumerate(image_paths, 1):
            print(f"   图像 {i}: {path}")

        # 卸载模型
        print("正在卸载模型...")
        generator.unload_model()

        return True

    except Exception as e:
        print(f"❌ 批量生成失败: {e}")
        return False


def test_model_info():
    """测试模型信息"""
    print("\n=== 测试 3: 模型信息 ===")

    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    except Exception as e:
        print(f"❌ 获取信息失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("文生图功能测试")
    print("=" * 60)

    results = []

    # 测试模型信息
    results.append(("模型信息", test_model_info()))

    # 测试简单生成
    results.append(("简单生成", test_simple_generation()))

    # 测试批量生成
    # results.append(("批量生成", test_batch_generation()))
    # 注意：批量生成测试已注释，因为可能需要较长时间

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{name}: {status}")

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()

