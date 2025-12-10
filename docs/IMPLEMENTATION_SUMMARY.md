# 文生图功能实现总结

## 已完成的工作

### 1. 核心模块实现 (`diffsynth/text_to_image.py`)

创建了完整的文生图生成器类 `TextToImageGenerator`，包含以下功能：

- **多模型支持**：
  - Stable Diffusion (SD 1.5/2.1)
  - Stable Diffusion XL (SDXL)
  - Stable Diffusion 3 (SD3)
  - Flux.1
  - HunyuanDiT

- **主要方法**：
  - `load_model()`: 加载指定类型的模型
  - `generate()`: 生成单张图像
  - `batch_generate()`: 批量生成图像
  - `unload_model()`: 卸载模型释放内存

- **便捷函数**：
  - `generate_image()`: 一键生成图像
  - `get_generator()`: 获取全局生成器实例

### 2. API 接口实现 (`controllers/text_to_image.py`)

创建了完整的 FastAPI REST API 端点：

- **POST /text-to-image/generate**: 生成单张图像
  - 请求参数：prompt, negative_prompt, model_type, width, height, num_inference_steps, guidance_scale, seed
  - 返回：图像路径

- **POST /text-to-image/batch-generate**: 批量生成图像
  - 支持多个提示词
  - 返回：图像路径列表

- **POST /text-to-image/load-model**: 预加载模型
  - 提前加载模型以加快生成速度

- **POST /text-to-image/unload-model**: 卸载模型
  - 释放显存和内存

### 3. Gradio UI 集成 (`gradio_app.py`)

在现有的 Gradio 界面中添加了"文生图"标签页：

- **用户界面**：
  - 提示词输入框（正向和负向）
  - 模型类型下拉选择
  - 图像尺寸滑块（宽度、高度）
  - 推理步数滑块
  - 引导系数滑块
  - 随机种子输入
  - 生成按钮

- **结果展示**：
  - 实时显示生成的图像
  - 显示生成信息（耗时、路径等）

### 4. 路由注册 (`controllers/route.py`)

将文生图 API 端点注册到主路由中，使其可以通过 FastAPI 应用访问。

### 5. 模块导出 (`diffsynth/__init__.py`)

导出了文生图相关的类和函数，方便其他模块使用。

### 6. 文档

创建了完整的文档：

- **docs/TEXT_TO_IMAGE.md**: 详细的使用文档
  - 安装说明
  - 使用方法（Gradio UI、Python API、REST API）
  - 参数说明
  - 高级用法
  - 性能优化建议
  - 故障排除

### 7. 测试脚本 (`test/test_text_to_image.py`)

创建了测试脚本，包含：
- 简单图像生成测试
- 批量生成测试
- 模型信息检查

## 文件结构

```
aduib-diffsynth/
├── diffsynth/
│   ├── __init__.py              # ✅ 已更新：导出文生图功能
│   ├── text_to_image.py         # ✅ 新增：文生图核心实现
│   ├── diffsynth.py             # 已存在：PDF处理
│   ├── aduib_ai.py              # 已存在：AduibAI客户端
│   └── mineru.py                # 已存在：PDF解析
│
├── controllers/
│   ├── __init__.py
│   ├── route.py                 # ✅ 已更新：注册文生图路由
│   ├── text_to_image.py         # ✅ 新增：文生图API端点
│   └── auth/
│
├── gradio_app.py                # ✅ 已更新：添加文生图UI
│
├── docs/
│   └── TEXT_TO_IMAGE.md         # ✅ 新增：完整使用文档
│
├── test/
│   ├── test_text_to_image.py    # ✅ 新增：测试脚本
│   └── ...
│
└── outputs/
    └── text_to_image/           # 自动创建：图像输出目录
```

## 功能特性

### 参数控制
- ✅ 提示词（正向/负向）
- ✅ 模型选择（5种模型类型）
- ✅ 图像尺寸（256-2048px）
- ✅ 推理步数（1-100步）
- ✅ 引导系数（1.0-20.0）
- ✅ 随机种子（可重现）

### 接口支持
- ✅ Gradio Web UI
- ✅ Python API
- ✅ FastAPI REST API

### 高级功能
- ✅ 批量生成
- ✅ 模型预加载
- ✅ 模型卸载
- ✅ 自定义输出路径
- ✅ 错误处理和日志

## 使用示例

### 1. Gradio UI
```bash
python gradio_app.py
# 访问 http://localhost:7860
# 选择"文生图"标签页
```

### 2. Python API

```python
from diffsynths.text_to_image import generate_image

image_path = generate_image(
    prompt="a beautiful sunset over mountains",
    model_type="sd",
    width=512,
    height=512,
)
```

### 3. REST API
```bash
curl -X POST "http://localhost:8000/api/text-to-image/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a beautiful sunset", "model_type": "sd"}'
```

## 依赖要求

### 必需依赖
- `diffsynth-engine`: 核心生成引擎
- `torch`: PyTorch（建议使用CUDA版本）
- `fastapi`: API框架
- `gradio`: Web UI框架
- `pydantic`: 数据验证

### 可选依赖
在 `pyproject.toml` 中已配置：
```toml
[project.optional-dependencies]
diffsynth=["diffsynth-engine"]
cuda=["torch==2.7.1","torchvision==0.22.1"]
```

安装命令：
```bash
uv sync --extra diffsynth --extra cuda
```

## 性能考虑

### 显存需求
- SD 1.5: ~4GB VRAM (512x512)
- SDXL: ~8GB VRAM (1024x1024)
- SD3/Flux: ~10GB+ VRAM

### 生成速度
- 取决于硬件（GPU/CPU）
- 推理步数影响时间
- 图像尺寸影响时间

### 优化建议
1. 使用合适的模型类型
2. 调整图像尺寸
3. 减少推理步数
4. 批量生成时复用模型
5. 生成完成后卸载模型

## 下一步扩展建议

### 可能的功能扩展
1. **图生图 (Image-to-Image)**
   - 基于输入图像生成新图像
   - 支持风格迁移

2. **图像编辑 (Inpainting)**
   - 局部修复/编辑图像
   - Mask 编辑功能

3. **ControlNet 支持**
   - 边缘检测控制
   - 姿态控制
   - 深度图控制

4. **LoRA 支持**
   - 加载自定义 LoRA 模型
   - 动态切换 LoRA

5. **图像上传管理**
   - 集成存储服务
   - 图像浏览和管理

6. **队列系统**
   - 支持异步生成
   - 任务队列管理

7. **用户历史**
   - 保存生成历史
   - 参数复用

## 测试清单

- ✅ 模块导入测试
- ✅ 简单生成测试
- ✅ 参数验证测试
- ✅ API 端点测试
- ✅ Gradio UI 测试
- ⏸️ 批量生成测试（耗时较长）
- ⏸️ 不同模型类型测试（需要下载模型）
- ⏸️ 错误处理测试

## 已知问题

1. IDE 可能显示 router 导入警告，但运行时正常
2. 首次加载模型时间较长
3. 需要下载对应的模型文件

## 总结

✅ 文生图功能已完整实现并集成到项目中：
- 核心生成引擎
- REST API 接口
- Gradio Web UI
- 完整文档
- 测试脚本

用户现在可以通过三种方式使用文生图功能：
1. Web UI（最简单）
2. Python API（最灵活）
3. REST API（最通用）

所有代码都遵循项目现有的架构和编码规范。

