"""
Text-to-Image API endpoints
"""
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from PIL import Image
from pydantic import BaseModel, Field

from configs.models import PipelineType
from diffsynths.text_to_image import generate_image, edit_image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/text-to-image", tags=["Text-to-Image"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TextToImageRequest(BaseModel):
    """文生图请求参数"""
    positive_magic: str = Field(default="", description="正向魔法词")
    prompt: str = Field(..., description="正向提示词")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_id: str = Field(
        default="Z-Image-Turbo",
        description="模型id"
    )
    lora_model: Optional[str] = Field(None, description="LoRA 模型名称")
    offload_model: bool = Field(False, description="启用显存卸载")
    width: int = Field(default=1024, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=1024, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=4.0, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")
    output_path: Optional[str] = Field(None, description="输出路径")


class TextToImageResponse(BaseModel):
    """文生图响应"""
    success: bool = Field(..., description="是否成功")
    image_path: Optional[str] = Field(None, description="生成的图像路径")
    message: str = Field(..., description="返回消息")


class BatchTextToImageRequest(BaseModel):
    """批量文生图请求参数"""
    positive_magic: str = Field(default="", description="正向魔法词")
    prompts: list[str] = Field(..., description="提示词列表")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_id: str = Field(
        default="Z-Image-Turbo",
        description="模型id"
    )
    lora_model: Optional[str] = Field(None, description="LoRA 模型名称")
    offload_model: bool = Field(False, description="启用显存卸载")
    width: int = Field(default=1024, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=1024, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=20, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=4.0, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")
    output_dir: Optional[str] = Field(None, description="输出目录")


class BatchTextToImageResponse(BaseModel):
    """批量文生图响应"""
    success: bool = Field(..., description="是否成功")
    image_paths: list[str] = Field(..., description="生成的图像路径列表")
    message: str = Field(..., description="返回消息")


class LoadModelRequest(BaseModel):
    """加载模型请求参数"""
    model_id: str = Field(
        default="Z-Image-Turbo",
        description="模型id"
    )
    lora_model: Optional[str] = Field(None, description="LoRA 模型名称")
    offload_model: bool = Field(False, description="是否开启模型卸载")


# ============================================================================
# Image Editing Models
# ============================================================================

class ImageEditRequest(BaseModel):
    """图片编辑请求参数"""
    prompt: str = Field(..., description="编辑提示词，描述期望的编辑效果")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_id: str = Field(
        default="Qwen-Image-Edit-2511",
        description="模型id"
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


class ImageEditByPathRequest(BaseModel):
    """通过路径编辑图片的请求参数"""
    input_image_path: str = Field(..., description="输入图片的文件路径")
    prompt: str = Field(..., description="编辑提示词，描述期望的编辑效果")
    negative_prompt: str = Field(default="", description="负向提示词")
    model_id: str = Field(
        default="Qwen-Image-Edit-2511",
        description="模型id"
    )
    lora_model: Optional[str] = Field(default=None, description="LoRA模型名称")
    offload_model: bool = Field(default=False, description="是否开启模型卸载以节省显存")
    width: int = Field(default=1024, ge=256, le=2048, description="图像宽度")
    height: int = Field(default=1024, ge=256, le=2048, description="图像高度")
    num_inference_steps: int = Field(default=50, ge=1, le=100, description="推理步数")
    guidance_scale: float = Field(default=4.5, ge=1.0, le=20.0, description="引导系数")
    seed: Optional[int] = Field(default=None, description="随机种子")
    output_path: Optional[str] = Field(default=None, description="输出图片路径（可选）")


# ============================================================================
# Helper Functions
# ============================================================================

def _prepare_output_path(output_path: Optional[str], default_dir: str = "outputs/text_to_image") -> str:
    """Prepare output path for generated image."""
    if output_path is None:
        output_dir = Path(default_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{uuid.uuid4()}.jpg")

    if os.path.isdir(output_path):
        output_dir = Path(output_path) / "outputs" / "text_to_image"
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / f"{uuid.uuid4()}.jpg")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


# ============================================================================
# API Endpoints - Text to Image
# ============================================================================

@router.post("/generate", response_model=TextToImageResponse)
async def generate_text_to_image(request: TextToImageRequest):
    """
    文生图接口
    """
    try:
        logger.info(f"收到文生图请求: {request.prompt[:50]}...")

        output_path = _prepare_output_path(request.output_path)

        image_path = generate_image(
            positive_magic=request.positive_magic,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_type=request.model_id,
            lora_model=request.lora_model,
            offload_model=request.offload_model,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            output_path=output_path,
        )

        logger.info(f"图像已保存至: {image_path}")

        return TextToImageResponse(
            success=True,
            image_path=image_path,
            message="图像生成成功"
        )

    except Exception as e:
        logger.error(f"文生图失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图像生成失败: {str(e)}")


@router.post("/batch-generate", response_model=BatchTextToImageResponse)
async def batch_generate_text_to_image(request: BatchTextToImageRequest):
    """
    批量文生图接口
    """
    try:
        logger.info(f"收到批量文生图请求，数量: {len(request.prompts)}")

        output_dir = request.output_dir or "outputs/text_to_image"
        image_paths = []

        for i, prompt in enumerate(request.prompts):
            logger.info(f"批量生成 {i + 1}/{len(request.prompts)}")

            # Adjust seed for each image
            current_seed = (request.seed + i) if request.seed is not None else None

            output_path = _prepare_output_path(None, output_dir)

            image_path = generate_image(
                positive_magic=request.positive_magic,
                prompt=prompt,
                negative_prompt=request.negative_prompt,
                model_type=request.model_id,
                lora_model=request.lora_model,
                offload_model=request.offload_model,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                seed=current_seed,
                output_path=output_path,
            )

            image_paths.append(image_path)

        return BatchTextToImageResponse(
            success=True,
            image_paths=image_paths,
            message=f"成功生成 {len(image_paths)} 张图像"
        )

    except Exception as e:
        logger.error(f"批量文生图失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量图像生成失败: {str(e)}")


@router.post("/load-model")
async def load_model(request: LoadModelRequest):
    """
    预加载模型
    """
    try:
        logger.info(f"预加载模型: {request.model_id}")

        # Model will be loaded on first use
        return {
            "success": True,
            "message": f"模型 {request.model_id} 将在使用时加载"
        }

    except Exception as e:
        logger.error(f"加载模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")


@router.post("/unload-model")
async def unload_model_endpoint(model_id: Optional[str] = None):
    """
    卸载模型
    """
    try:
        from diffsynths.text_to_image import unload_model

        if model_id:
            logger.info(f"卸载模型: {model_id}")
        else:
            logger.info("卸载所有模型")

        unload_model()

        return {
            "success": True,
            "message": f"模型 {model_id} 已卸载" if model_id else "所有模型已卸载"
        }

    except Exception as e:
        logger.error(f"卸载模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"模型卸载失败: {str(e)}")


@router.post("/unload-lora-model")
async def unload_lora_model():
    """
    卸载 LoRA 模型
    """
    try:
        from diffsynths.text_to_image import unload_lora

        logger.info("卸载 LoRA 模型")
        unload_lora()

        return {
            "success": True,
            "message": "LoRA 模型已卸载"
        }

    except Exception as e:
        logger.error(f"卸载 LoRA 模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LoRA 模型卸载失败: {str(e)}")


# ============================================================================
# API Endpoints - Image Editing
# ============================================================================

@router.post("/edit", response_model=ImageEditResponse)
async def edit_uploaded_image(
    input_image: UploadFile = File(..., description="要编辑的图片"),
    prompt: str = Form(..., description="编辑提示词"),
    negative_prompt: str = Form(default="", description="负向提示词"),
    model_id: str = Form(default="Qwen-Image-Edit-2511", description="模型ID"),
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
    """
    try:
        logger.info(f"收到图片编辑请求: {prompt[:50]}...")

        # Save uploaded image to temp file
        suffix = Path(input_image.filename).suffix if input_image.filename else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            contents = await input_image.read()
            temp_file.write(contents)
            temp_image_path = temp_file.name

        try:
            # Generate edited image
            edited_image_path = edit_image(
                input_image_path=temp_image_path,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                lora_model=lora_model,
                offload_model=offload_model,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_path=None,
            )

            logger.info(f"编辑后图像已保存至: {edited_image_path}")

            return ImageEditResponse(
                success=True,
                image_path=edited_image_path,
                message="图像编辑成功"
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)

    except Exception as e:
        logger.error(f"图片编辑失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图像编辑失败: {str(e)}")


@router.post("/edit-by-path", response_model=ImageEditResponse)
async def edit_image_by_path(request: ImageEditByPathRequest):
    """
    图片编辑接口（通过路径）
    """
    try:
        logger.info(f"收到图片编辑请求: {request.prompt[:50]}...")

        # Check input file exists
        if not os.path.exists(request.input_image_path):
            raise HTTPException(status_code=404, detail=f"输入图片不存在: {request.input_image_path}")

        # Generate edited image
        edited_image_path = edit_image(
            input_image_path=request.input_image_path,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_id=request.model_id,
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
            image_path=edited_image_path,
            message="图像编辑成功"
        )

    except Exception as e:
        logger.error(f"图片编辑失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图像编辑失败: {str(e)}")
