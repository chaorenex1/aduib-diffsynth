# 模型管理功能实现说明

## 概述

本次更新为 Gradio 文生图界面添加了完整的模型和 LoRA 管理功能，包括：
- 模型状态显示
- LoRA 状态显示
- 卸载模型按钮
- 卸载 LoRA 按钮
- 刷新状态按钮
- 自动切换模型时卸载旧模型

## 主要更改

### 1. `diffsynths/text_to_image.py`

#### 新增功能
- **`unload_model()`**: 卸载当前加载的主模型
- **`unload_lora()`**: 卸载当前加载的 LoRA 模型
- **`get_model_status()`**: 获取当前模型和 LoRA 的加载状态

#### 类更新
**`TextToImageGenerator`** 类新增属性：
- `current_model_type`: 跟踪当前加载的模型类型
- 更新后的 `lora_loaded`: 跟踪 LoRA 是否已加载

#### 智能模型切换
- 在 `generate_image()` 函数中添加了智能检测
- 当用户选择不同的模型类型时，自动卸载旧模型
- 避免多个模型同时占用显存

### 2. `gradio_app.py`

#### UI 组件新增
在"文生图"标签页顶部添加：

1. **状态显示区域**
   - `model_status_text`: 显示主模型状态（已加载/未加载）
   - `lora_status_text`: 显示 LoRA 状态（已加载/未加载）

2. **控制按钮**
   - `unload_model_button`: 🗑️ 卸载模型
   - `unload_lora_button`: 🗑️ 卸载LoRA
   - `refresh_status_button`: 🔄 刷新状态

#### 回调函数

**`generate_image_gradio()`**
- 更新后会同时返回生成的图像和更新后的状态
- 输出包括：图像路径、生成信息、模型状态、LoRA状态

**`unload_model_gradio()`**
- 卸载主模型
- 同时清除 LoRA（因为 LoRA 依赖主模型）
- 返回更新后的状态和操作结果

**`unload_lora_gradio()`**
- 只卸载 LoRA 模型
- 保留主模型
- 返回更新后的状态和操作结果

**`refresh_status_gradio()`**
- 刷新并显示当前模型和 LoRA 的加载状态
- 不执行任何加载/卸载操作

## 使用场景

### 场景 1: 切换不同模型
1. 用户首次选择 "MusePublic/Qwen-image" 生成图像
2. 模型加载，状态显示 "✅ 已加载 (MusePublic/Qwen-image)"
3. 用户切换到 "Qwen-Image" 
4. 系统自动卸载旧模型，加载新模型
5. 状态更新为 "✅ 已加载 (Qwen-Image)"

### 场景 2: 只卸载 LoRA
1. 用户加载了模型和 LoRA
2. 点击 "🗑️ 卸载LoRA" 按钮
3. LoRA 被卸载，主模型保持加载状态
4. 可以继续使用主模型生成图像

### 场景 3: 完全卸载
1. 用户想释放所有显存
2. 点击 "🗑️ 卸载模型" 按钮
3. 主模型和 LoRA 都被卸载
4. 显存被释放
5. 状态显示 "❌ 未加载"

### 场景 4: 检查状态
1. 用户不确定当前模型状态
2. 点击 "🔄 刷新状态" 按钮
3. 显示最新的加载状态

## 技术细节

### 内存管理
- 卸载模型时会调用 `torch.cuda.empty_cache()` 清理 CUDA 缓存
- 使用 `del` 删除模型对象，触发垃圾回收

### 状态同步
- 所有生成操作后自动更新状态显示
- 状态使用 emoji 图标直观显示：
  - ✅ 已加载
  - ❌ 未加载
  - ⚠️ 状态未知（发生错误时）

### 错误处理
- 所有操作都包含 try-except 错误处理
- 失败时显示详细错误信息
- 错误后不影响其他功能继续使用

## 代码示例

### 直接使用 API
```python
from diffsynths.text_to_image import get_model_status, unload_model, unload_lora

# 检查状态
model_loaded, lora_loaded = get_model_status()
print(f"模型已加载: {model_loaded}, LoRA已加载: {lora_loaded}")

# 卸载 LoRA
unload_lora()

# 卸载模型
unload_model()
```

### 在 Gradio 中使用
用户只需点击界面上的按钮，所有操作都会自动执行。

## 注意事项

1. **LoRA 依赖主模型**: 卸载主模型会同时卸载 LoRA
2. **切换模型自动卸载**: 无需手动卸载，系统会自动处理
3. **显存释放**: 卸载模型后会立即释放显存
4. **状态实时更新**: 每次操作后状态都会自动刷新

## 测试建议

1. 测试加载不同模型类型
2. 测试加载/卸载 LoRA
3. 测试完全卸载模型
4. 测试模型切换时的自动卸载
5. 测试状态刷新功能
6. 测试错误处理（如在未加载时尝试卸载）

## 未来改进方向

1. 添加模型预热功能
2. 支持多模型并发加载
3. 添加显存使用情况监控
4. 支持模型缓存机制
5. 添加加载进度显示

