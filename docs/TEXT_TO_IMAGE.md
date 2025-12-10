# 文生图功能文档

## 概述

本项目已集成使用 DiffSynth Engine 的文生图功能，支持多种扩散模型进行图像生成。

## 功能特性

- 支持多种模型：Stable Diffusion (SD), SDXL, SD3, Flux, HunyuanDiT
- 灵活的参数配置：图像尺寸、推理步数、引导系数、随机种子等
- 批量生成支持
- Gradio Web UI 界面
- FastAPI REST API 接口
- 模型预加载和卸载功能

## 使用方法

### 1. 安装依赖

确保安装了 diffsynth-engine 和相关依赖：

```bash
# 安装 diffsynth-engine
uv pip install diffsynth-engine

# 如果使用 CUDA
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

或使用项目的可选依赖：

```bash
uv sync --extra diffsynth --extra cuda
```

### 2. Gradio Web UI 使用

启动 Gradio 应用：

```bash
python gradio_app.py
```

然后访问 http://localhost:7860，在"文生图"标签页中：

1. 输入正向提示词（描述想要生成的图像）
2. 输入负向提示词（描述不想出现的元素，可选）
3. 选择模型类型（sd/sdxl/sd3/flux/hunyuan）
4. 调整参数：
   - 宽度和高度（256-2048像素）
   - 推理步数（1-100步）
   - 引导系数（1.0-20.0）
   - 随机种子（-1表示随机）
5. 点击"🎨 生成图像"按钮

### 3. Python API 使用

```python
from diffsynths.text_to_image import generate_image

# 简单使用
image_path = generate_image(
    prompt="a beautiful sunset over mountains",
    negative_prompt="low quality, blurry",
    model_type="sd",
    width=512,
    height=512,
)

print(f"图像已保存至: {image_path}")
```

### 4. REST API 使用

启动 FastAPI 服务：

```bash
python app.py
```

#### 4.1 生成单张图像

```bash
curl -X POST "http://localhost:8000/api/text-to-image/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a beautiful sunset over mountains",
    "negative_prompt": "low quality, blurry",
    "model_type": "sd",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  }'
```

响应：

```json
{
  "success": true,
  "image_path": "outputs/text_to_image/xxxxx.png",
  "message": "图像生成成功"
}
```

#### 4.2 批量生成图像

```bash
curl -X POST "http://localhost:8000/api/text-to-image/batch-generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "a red apple",
      "a blue car",
      "a green tree"
    ],
    "model_type": "sd",
    "width": 512,
    "height": 512
  }'
```

#### 4.3 预加载模型

```bash
curl -X POST "http://localhost:8000/api/text-to-image/load-model?model_type=sdxl"
```

#### 4.4 卸载模型

```bash
curl -X POST "http://localhost:8000/api/text-to-image/unload-model"
```

## 高级用法

### 批量生成

```python
from diffsynths.text_to_image import get_generator

generator = get_generator()
generator.load_model("sd")

prompts = [
    "a red apple on a table",
    "a blue car in the street",
    "a green tree in the park",
]

image_paths = generator.batch_generate(
    prompts=prompts,
    width=512,
    height=512,
    num_inference_steps=20,
    guidance_scale=7.5,
    seed=42,  # 每张图会使用 seed, seed+1, seed+2...
)

for i, path in enumerate(image_paths):
    print(f"图像 {i + 1} 已保存至: {path}")
```

### 使用自定义模型

```python
from diffsynths.text_to_image import TextToImageGenerator

generator = TextToImageGenerator(
    model_path="/path/to/your/model.safetensors",
    device="cuda"
)

generator.load_model("sd")

image_path = generator.generate(
    prompt="your prompt here",
    width=768,
    height=768,
)
```

## 参数说明

### 模型类型 (model_type)

- `sd`: Stable Diffusion 1.5/2.1
- `sdxl`: Stable Diffusion XL
- `sd3`: Stable Diffusion 3
- `flux`: Flux.1
- `hunyuan`: HunyuanDiT

### 提示词 (prompt)

描述您想要生成的图像内容。建议：
- 使用清晰、具体的描述
- 可以包含风格、光照、细节等信息
- 英文效果通常更好

示例：
```
"a beautiful landscape with mountains and a lake, sunset, 4k, highly detailed"
```

### 负向提示词 (negative_prompt)

描述您不想在图像中出现的元素。常用：
```
"low quality, blurry, deformed, ugly, bad anatomy, watermark"
```

### 图像尺寸 (width, height)

- 范围：256-2048 像素
- 建议使用 64 的倍数
- SD 1.5: 512x512
- SDXL: 1024x1024
- 更大尺寸需要更多显存

### 推理步数 (num_inference_steps)

- 范围：1-100
- 推荐：20-50
- 步数越多，质量越好，但生成时间越长

### 引导系数 (guidance_scale/cfg_scale)

- 范围：1.0-20.0
- 推荐：7.0-10.0
- 越高越贴近提示词，但可能过度饱和
- 越低越有创意，但可能偏离提示词

### 随机种子 (seed)

- 可选参数
- 使用相同种子和参数可以重现相同图像
- 设置为 -1 或 None 使用随机种子

## 目录结构

```
diffsynth/
├── __init__.py              # 模块导出
├── text_to_image.py         # 文生图核心实现
├── diffsynth.py             # PDF 处理功能
├── aduib_ai.py              # AduibAI 客户端
└── mineru.py                # MinerU PDF 解析

controllers/
├── text_to_image.py         # 文生图 API 端点
└── route.py                 # API 路由注册

outputs/
└── text_to_image/           # 生成的图像输出目录
    ├── xxxxx.png
    └── ...
```

## 性能优化建议

1. **显存管理**：
   - 使用较小的图像尺寸（512x512）
   - 减少推理步数（20步）
   - 生成完成后卸载模型释放显存

2. **批量生成**：
   - 使用 `batch_generate` 方法可以复用已加载的模型
   - 避免重复加载模型

3. **模型选择**：
   - SD 1.5: 最快，显存占用最小
   - SDXL: 质量更好，但更慢
   - Flux/SD3: 最新技术，质量最佳

## 故障排除

### 导入错误

```
ImportError: cannot import name 'ModelManager' from 'diffsynth'
```

解决：安装 diffsynth-engine
```bash
uv pip install diffsynth-engine
```

### CUDA 内存不足

```
RuntimeError: CUDA out of memory
```

解决：
- 减小图像尺寸
- 减少推理步数
- 卸载其他模型
- 使用 CPU（会很慢）

### 模型文件未找到

确保模型文件在正确位置，或提供正确的 `model_path` 参数。

## 示例代码

完整示例请参考：
- `diffsynth/text_to_image.py` - 核心实现
- `controllers/text_to_image.py` - API 端点
- `gradio_app.py` - Web UI 集成

## 支持

如有问题，请查看：
- [DiffSynth Engine 官方文档](https://github.com/modelscope/DiffSynth-Studio)
- 项目 Issue 页面

