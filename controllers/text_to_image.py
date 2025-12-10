"""
Text-to-Image API endpoints
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from diffsynths.text_to_image import generate_image, get_generator, edit_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text-to-image", tags=["Text-to-Image"])


class TextToImageRequest(BaseModel):
    """文生图请求参数"""
    positive_magic:str=Field(default="",description="正向魔法词")
    prompt: str = Field(..., description="正向提示词")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_type: str = Field(
        default="sd", description="模型类型"
    )
    width: int = Field(default=512, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=512, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")


class TextToImageResponse(BaseModel):
    """文生图响应"""
    success: bool = Field(..., description="是否成功")
    image_path: Optional[str] = Field(None, description="生成的图像路径")
    message: str = Field(..., description="返回消息")


@router.post("/generate", response_model=TextToImageResponse)
async def generate_text_to_image(request: TextToImageRequest):
    """
    文生图接口

    生成图像并返回图像路径
    """
    try:
        logger.info(f"收到文生图请求: {request.prompt[:50]}...")

        # 生成图像
        image_path = generate_image(
            positive_magic=request.positive_magic,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_type=request.model_type,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )

        return TextToImageResponse(
            success=True,
            image_path=image_path,
            message="图像生成成功"
        )

    except Exception as e:
        logger.error(f"文生图失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图像生成失败: {str(e)}")


class BatchTextToImageRequest(BaseModel):
    """批量文生图请求参数"""
    positive_magic:str = Field(default="", description="正向魔法词")
    prompts: list[str] = Field(..., description="提示词列表")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_type: str = Field(
        default="sd", description="模型类型"
    )
    width: int = Field(default=512, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=512, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")


class BatchTextToImageResponse(BaseModel):
    """批量文生图响应"""
    success: bool = Field(..., description="是否成功")
    image_paths: list[str] = Field(..., description="生成的图像路径列表")
    message: str = Field(..., description="返回消息")


@router.post("/batch-generate", response_model=BatchTextToImageResponse)
async def batch_generate_text_to_image(request: BatchTextToImageRequest):
    """
    批量文生图接口

    根据提示词列表批量生成图像
    """
    try:
        logger.info(f"收到批量文生图请求，数量: {len(request.prompts)}")

        generator = get_generator()

        # 加载模型（如果还未加载）
        if generator.pipe is None:
            generator.load_model(request.model_type)

        # 批量生成
        image_paths = generator.batch_generate(
            positive_magic=request.positive_magic,
            prompts=request.prompts,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )

        return BatchTextToImageResponse(
            success=True,
            image_paths=image_paths,
            message=f"成功生成 {len(image_paths)} 张图像"
        )

    except Exception as e:
        logger.error(f"批量文生图失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量图像生成失败: {str(e)}")


class LoadModelRequest(BaseModel):
    """加载模型请求参数"""
    model_type: str = Field(
        default="sd", description="模型类型"
    )
    lora_model: str = Field(default="", description="lora模型名称")
    offload_model: bool = Field(default=False, description="是否开启模型卸载")


@router.post("/load-model")
async def load_model(request: LoadModelRequest):
    """
    预加载模型

    提前加载模型以加快后续生成速度
    """
    try:
        logger.info(f"预加载模型: {request.model_type}")
        generator = get_generator()
        generator.load_model(request.model_type, lora_model=request.lora_model, low_varam=request.offload_model)

        return {
            "success": True,
            "message": f"模型 {request.model_type} 加载成功"
        }

    except Exception as e:
        logger.error(f"加载模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@router.post("/unload-model")
async def unload_model():
    """
    卸载模型

    释放显存和内存
    """
    try:
        logger.info("卸载模型")
        generator = get_generator()
        generator.unload_model()

        return {
            "success": True,
            "message": "模型已卸载"
        }

    except Exception as e:
        logger.error(f"卸载模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型卸载失败: {str(e)}")


@router.post("/unload-lora-model")
async def unload_lora_model():
    """
    卸载 LoRA 模型

    释放显存和内存
    """
    try:
        logger.info("卸载 LoRA 模型")
        generator = get_generator()
        generator.unload_model()

        return {
            "success": True,
            "message": "LoRA 模型已卸载"
        }

    except Exception as e:
        logger.error(f"卸载 LoRA 模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LoRA 模型卸载失败: {str(e)}")


# ==================== 图片编辑接口 ====================

class ImageEditRequest(BaseModel):
    """图片编辑请求参数"""
    prompt: str = Field(..., description="编辑提示词，描述期望的编辑效果")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_type: str = Field(
        default="Qwen-Image-Edit-2509", description="模型类型"
    )
    lora_model: Optional[str] = Field(default=None, description="LoRA模型名称")
    offload_model: bool = Field(default=False, description="是否开启模型卸载以节省显存")
    width: int = Field(default=1024, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=1024, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=50, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=4.5, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")


class ImageEditResponse(BaseModel):
    """图片编辑响应"""
    success: bool = Field(..., description="是否成功")
    image_path: Optional[str] = Field(None, description="编辑后的图像路径")
    message: str = Field(..., description="返回消息")


@router.post("/edit", response_model=ImageEditResponse)
async def edit_uploaded_image(
    input_image: UploadFile = File(..., description="要编辑的图片"),
    prompt: str = Form(..., description="编辑提示词"),
    negative_prompt: str = Form(default="", description="负向提示词"),
    model_type: str = Form(default="Qwen-Image-Edit-2509", description="模型类型"),
    lora_model: Optional[str] = Form(default=None, description="LoRA模型名称"),
    offload_model: bool = Form(default=False, description="是否开启模型卸载"),
    width: int = Form(default=1024, description="图像宽度"),
    height: int = Form(default=1024, description="图像高度"),
    num_inference_steps: int = Form(default=50, description="推理步数"),
    guidance_scale: float = Form(default=4.5, description="引导系数"),
    seed: Optional[int] = Form(default=None, description="随机种子"),
):
    """
    图片编辑接口

    上传图片并根据提示词进行编辑，返回编辑后的图像路径
    """
    import tempfile
    import os
    from pathlib import Path

    try:
        logger.info(f"收到图片编辑请求: {prompt[:50]}...")

        # 保存上传的图片到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(input_image.filename).suffix) as temp_file:
            contents = await input_image.read()
            temp_file.write(contents)
            temp_image_path = temp_file.name

        try:
            # 编辑图像
            output_path = edit_image(
                input_image_path=temp_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_type=model_type,
                lora_model=lora_model,
                offload_model=offload_model,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )

            return ImageEditResponse(
                success=True,
                image_path=output_path,
                message="图像编辑成功"
            )
        finally:
            # 清理临时文件
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        logger.error(f"图片编辑失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图像编辑失败: {str(e)}")


class ImageEditByPathRequest(BaseModel):
    """通过路径编辑图片的请求参数"""
    input_image_path: str = Field(..., description="输入图片的文件路径")
    prompt: str = Field(..., description="编辑提示词，描述期望的编辑效果")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_type: str = Field(
        default="Qwen-Image-Edit-2509", description="模型类型"
    )
    lora_model: Optional[str] = Field(default=None, description="LoRA模型名称")
    offload_model: bool = Field(default=False, description="是否开启模型卸载以节省显存")
    width: int = Field(default=1024, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=1024, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=50, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=4.5, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")
    output_path: Optional[str] = Field(default=None, description="输出图片路径（可选）")


@router.post("/edit-by-path", response_model=ImageEditResponse)
async def edit_image_by_path(request: ImageEditByPathRequest):
    """
    图片编辑接口（通过路径）

    根据已有图片路径和提示词进行编辑，返回编辑后的图像路径
    """
    import os

    try:
        logger.info(f"收到图片编辑请求: {request.prompt[:50]}...")

        # 检查输入文件是否存在
        if not os.path.exists(request.input_image_path):
            raise HTTPException(status_code=404, detail=f"输入图片不存在: {request.input_image_path}")

        # 编辑图像
        output_path = edit_image(
            input_image_path=request.input_image_path,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_type=request.model_type,
            lora_model=request.lora_model,
            offload_model=request.offload_model,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=request.output_path,
        )

        return ImageEditResponse(
            success=True,
            image_path=output_path,
            message="图像编辑成功"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"图片编辑失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图像编辑失败: {str(e)}")

