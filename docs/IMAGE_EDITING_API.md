# 图片编辑API文档

## 概述

图片编辑API提供了两个接口，用于对已有图片进行AI编辑。这些接口基于 Qwen-Image-Edit-2509 模型，可以根据文本提示词对图片进行编辑。

## API端点

### 1. 上传图片编辑 (POST /text-to-image/edit)

上传图片并根据提示词进行编辑。

#### 请求参数

**Content-Type:** `multipart/form-data`

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|-------|------|------|-------|------|
| input_image | File | 是 | - | 要编辑的图片文件 |
| prompt | string | 是 | - | 编辑提示词，描述期望的编辑效果 |
| negative_prompt | string | 否 | "" | 负向提示词 |
| model_type | string | 否 | "Qwen-Image-Edit-2509" | 模型类型 |
| lora_model | string | 否 | null | LoRA模型名称 |
| offload_model | boolean | 否 | false | 是否开启模型卸载以节省显存 |
| width | integer | 否 | 1024 | 图像宽度 (256-2048) |
| height | integer | 否 | 1024 | 图像高度 (256-2048) |
| num_inference_steps | integer | 否 | 50 | 推理步数 (1-100) |
| guidance_scale | float | 否 | 4.5 | 引导系数 (1.0-20.0) |
| seed | integer | 否 | null | 随机种子 |

#### 响应示例

```json
{
  "success": true,
  "image_path": "outputs/image_edit/12345678-1234-5678-1234-567812345678.png",
  "message": "图像编辑成功"
}
```

#### cURL示例

```bash
curl -X POST "http://localhost:8000/text-to-image/edit" \
  -H "Content-Type: multipart/form-data" \
  -F "input_image=@/path/to/your/image.jpg" \
  -F "prompt=把天空变成日落的颜色" \
  -F "negative_prompt=模糊,低质量" \
  -F "width=1024" \
  -F "height=1024" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=4.5"
```

#### Python示例

```python
import requests

url = "http://localhost:8000/text-to-image/edit"

# 准备文件和数据
files = {
    'input_image': open('image.jpg', 'rb')
}

data = {
    'prompt': '把天空变成日落的颜色',
    'negative_prompt': '模糊,低质量',
    'width': 1024,
    'height': 1024,
    'num_inference_steps': 50,
    'guidance_scale': 4.5
}

response = requests.post(url, files=files, data=data)
result = response.json()
print(f"编辑后的图片路径: {result['image_path']}")
```

---

### 2. 通过路径编辑图片 (POST /text-to-image/edit-by-path)

使用服务器上已存在的图片路径进行编辑。

#### 请求参数

**Content-Type:** `application/json`

```json
{
  "input_image_path": "string (必填) - 输入图片的文件路径",
  "prompt": "string (必填) - 编辑提示词",
  "negative_prompt": "string (可选) - 负向提示词",
  "model_type": "string (可选,默认: Qwen-Image-Edit-2509) - 模型类型",
  "lora_model": "string (可选) - LoRA模型名称",
  "offload_model": "boolean (可选,默认: false) - 是否开启模型卸载",
  "width": "integer (可选,默认: 1024) - 图像宽度",
  "height": "integer (可选,默认: 1024) - 图像高度",
  "num_inference_steps": "integer (可选,默认: 50) - 推理步数",
  "guidance_scale": "float (可选,默认: 4.5) - 引导系数",
  "seed": "integer (可选) - 随机种子",
  "output_path": "string (可选) - 输出图片路径"
}
```

#### 响应示例

```json
{
  "success": true,
  "image_path": "outputs/image_edit/12345678-1234-5678-1234-567812345678.png",
  "message": "图像编辑成功"
}
```

#### cURL示例

```bash
curl -X POST "http://localhost:8000/text-to-image/edit-by-path" \
  -H "Content-Type: application/json" \
  -d '{
    "input_image_path": "/path/to/server/image.jpg",
    "prompt": "把天空变成日落的颜色",
    "negative_prompt": "模糊,低质量",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.5
  }'
```

#### Python示例

```python
import requests

url = "http://localhost:8000/text-to-image/edit-by-path"

payload = {
    "input_image_path": "/path/to/server/image.jpg",
    "prompt": "把天空变成日落的颜色",
    "negative_prompt": "模糊,低质量",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 4.5
}

response = requests.post(url, json=payload)
result = response.json()
print(f"编辑后的图片路径: {result['image_path']}")
```

---

## 模型管理

图片编辑功能复用了文生图API的模型管理接口：

### 加载模型

```bash
POST /text-to-image/load-model
```

### 卸载模型

```bash
POST /text-to-image/unload-model
```

### 卸载LoRA模型

```bash
POST /text-to-image/unload-lora-model
```

---

## 使用建议

### 1. 提示词编写

- **明确描述编辑目标**：如"把天空变成日落的颜色"、"给人物添加太阳镜"
- **使用负向提示词**：避免不想要的效果，如"模糊,低质量,变形"
- **分步骤编辑**：对于复杂的编辑，可以分多次进行

### 2. 参数调优

- **guidance_scale (引导系数)**：
  - 较低值 (3.0-5.0)：更自然，但可能偏离提示词
  - 较高值 (5.0-8.0)：更贴合提示词，但可能过度编辑
  
- **num_inference_steps (推理步数)**：
  - 20-30步：快速预览
  - 50步：标准质量（推荐）
  - 80-100步：最高质量，但速度较慢

- **width/height (图像尺寸)**：
  - 建议保持原图比例
  - 使用1024x1024可获得最佳效果
  - 较大尺寸需要更多显存

### 3. 性能优化

- **offload_model**：如果显存不足，可以开启模型卸载
- **预加载模型**：批量编辑前可以预先加载模型
- **batch处理**：如需编辑多张图片，可以保持模型加载状态

---

## 错误处理

### 常见错误

| 错误码 | 说明 | 解决方案 |
|-------|------|---------|
| 404 | 输入图片不存在 | 检查图片路径是否正确 |
| 500 | 图像编辑失败 | 检查模型是否正确加载，参数是否有效 |

### 错误响应示例

```json
{
  "detail": "输入图片不存在: /path/to/image.jpg"
}
```

---

## 支持的图片格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- WebP (.webp)

输出格式固定为PNG格式，以保证最佳质量。

---

## 注意事项

1. 上传的图片会被自动调整为指定的宽度和高度
2. 编辑后的图片默认保存在 `outputs/image_edit/` 目录
3. 首次使用会自动下载模型，可能需要较长时间
4. 建议先使用较少的推理步数进行测试，确认效果后再使用高步数
5. 如果遇到显存不足，可以尝试减小图片尺寸或开启 `offload_model`

---

## 更新日志

### v1.0.0 (2025-12-10)
- ✨ 新增图片编辑功能
- ✨ 支持上传图片编辑
- ✨ 支持通过路径编辑图片
- ✨ 集成 Qwen-Image-Edit-2509 模型

