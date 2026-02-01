"""
Text-to-Image generation using DiffSynth Engine
"""
import logging
import os
import traceback
from pathlib import Path
from typing import Optional

from PIL import Image
from modelscope import snapshot_download

from configs.models import get_model_config, PipelineType, PipelineClass, ModelConfig

logger = logging.getLogger(__name__)

import torch

def get_gpu_devices():
    if not torch.cuda.is_available():
        return "cpu"

    count = torch.cuda.device_count()
    devices =["cuda:"+str(i) for i in range(count)]
    logger.debug(f"检测到可用GPU设备: {devices}")
    return "cuda"



class TextToImageGenerator:
    """文生图生成器，使用 DiffSynth Engine"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        初始化文生图生成器
        
        参数:
            model_path: 模型路径，如果为None则使用默认模型
            device: 运行设备，cuda或cpu
        """
        self.current_model_id = None
        self.device = device
        self.model_path = model_path
        self.pipe = None
        self.lora_loaded=False
        if not os.path.exists(model_path):
            os.makedirs(model_path,exist_ok=True)
        logger.info(f"初始化 TextToImageGenerator，设备: {get_gpu_devices()}")
        
    def load_model(
        self,
        pipeline_type: str = PipelineType.TEXT_TO_IMAGE.value,
        model_id: str = "MusePublic/Qwen-image",
        lora_model:str=None,
        low_varam: bool = False,
    ):
        """
        加载模型
        
        参数:
            model_id: 模型类型，支持 sd, sdxl, sd3, flux, hunyuan
        """
        try:
            logger.info(f"load model: {model_id}")
            model_config = get_model_config(pipeline_type, model_id)
            pipeline_class = model_config.pipeline
            match pipeline_class:
                case PipelineClass.QWEN_IMAGE.value:
                     self.pipe=self._load_qwen_image(pipeline_type,model_id, low_varam)
                case PipelineClass.Z_IMAGE.value:
                    self.pipe=self._load_z_image(pipeline_type,model_id, low_varam)
                case PipelineClass.FLUX2_IMAGE.value:
                    self.pipe=self._load_flux2_image(pipeline_type,model_id, low_varam)
                case _:
                    raise ValueError(f"Unsupported model type: {model_id}")


            if lora_model:
                self.load_lora_model( lora_model)
                self.lora_loaded=True
                logger.info(f"LoRA模型加载完成: {lora_model}")
        # if low_varam:
        #     logger.info("启用显存管理模式...")
        #     self.pipe.enable_vram_management()


            logger.info(f"模型加载完成: {model_id}")
            
        except ImportError as e:
            logger.error(f"导入 diffsynth 模块失败: {e}")
            logger.error("请确保已安装 diffsynth-engine: pip install diffsynth-engine")
            raise
        except Exception as e:
            traceback.print_exc()
            logger.error(f"加载模型失败: {e}")
            raise

    def _load_qwen_image(self,
                         pipeline_type: str,
                         model_id: str,
                         offload_model: bool):
        """Load MusePublic/Qwen-image model."""
        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
        config = get_model_config(pipeline_type, model_id)

        self.download_models(config, model_id)

        vram = {}
        if offload_model:
            vram ={
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

        return QwenImagePipeline.from_pretrained(
            torch_dtype=config.d_type.get_torch_dtype(),
            device=self.device,
            model_configs=[
                ModelConfig(
                    model_id=model_id,
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
                ModelConfig(
                    model_id=config.text_encoder if config.text_encoder else model_id,
                    origin_file_pattern="text_encoder/model*.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
                ModelConfig(
                    model_id=config.vae_encoder if config.vae_encoder else model_id,
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id=config.tokenizer if config.tokenizer else model_id,
                origin_file_pattern="tokenizer/",
                skip_download=True,
                local_model_path=self.model_path,
            ) if pipeline_type == PipelineType.TEXT_TO_IMAGE.value else None,
            processor_config=ModelConfig(
                model_id=config.tokenizer if config.tokenizer else model_id,
                origin_file_pattern="processor/",
                skip_download=True,
                local_model_path=self.model_path,
            ) if pipeline_type == PipelineType.IMAGE_EDIT.value else None,
        )

    def _load_z_image(self,
                            pipeline_type: str,
                            model_id: str,
                            offload_model: bool):
        """Load Tongyi-MAI/Z-Image-Turbo model."""
        from diffsynth.pipelines.z_image import ZImagePipeline, ModelConfig
        config = get_model_config(pipeline_type, model_id)

        self.download_models(config, model_id)

        vram = {}
        if offload_model:
            vram = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

        return ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(
                    model_id=model_id,
                    origin_file_pattern="transformer/*.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
                ModelConfig(
                    model_id=config.text_encoder if config.text_encoder else model_id,
                    origin_file_pattern="text_encoder/*.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
                ModelConfig(
                    model_id=config.vae_encoder if config.vae_encoder else model_id,
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id=config.tokenizer if config.tokenizer else model_id,
                origin_file_pattern="tokenizer/",
                skip_download=True,
                local_model_path=self.model_path,
                **vram
            ),
        )

    def _load_flux2_image(self,pipeline_type: str,
                            model_id: str,
                            offload_model: bool):
        """Load Flux model."""
        from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
        config = get_model_config(pipeline_type, model_id)

        self.download_models(config, model_id)

        vram = {}
        if offload_model:
            vram = {
    "offload_dtype": "disk",
    "offload_device": "disk",
    "onload_dtype": torch.float8_e4m3fn,
    "onload_device": "cpu",
    "preparing_dtype": torch.float8_e4m3fn,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

        return Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(
                    model_id=model_id,
                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
                ModelConfig(
                    model_id=config.text_encoder if config.text_encoder else model_id,
                    origin_file_pattern="text_encoder/model*.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
                ModelConfig(
                    model_id=config.vae_encoder if config.vae_encoder else model_id,
                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                    skip_download=True,
                    local_model_path=self.model_path,
                    **vram
                ),
            ],
            tokenizer_config=ModelConfig(
                model_id=config.tokenizer if config.tokenizer else model_id,
                origin_file_pattern="tokenizer/",
                skip_download=True,
                local_model_path=self.model_path,
            ),
        )

    def load_lora_model(self, lora_model: str):
        """Load LoRA model into the pipeline."""
        from diffsynth.pipelines.z_image import ModelConfig
        if lora_model:
            lora_config = ModelConfig(model_id=lora_model,
                                      origin_file_pattern="*",
                                      skip_download=True,
                                      local_model_path=self.model_path)
            if not os.path.exists(os.path.join(self.model_path, lora_model)):
                snapshot_download(lora_model, local_dir=self.model_path,
                                  allow_file_pattern="*")

            self.pipe.load_lora(self.pipe.dit, lora_config=lora_config)
            logger.info(f"LoRA模型加载完成: {lora_model}")


    def download_models(self, config: ModelConfig, model_id: str):
        model_path = os.path.join(self.model_path, model_id)
        if not os.path.exists(model_path):
            snapshot_download(model_id, local_dir=model_path, allow_file_pattern="*")

        if config.text_encoder:
            text_encoder_model_path = os.path.join(self.model_path, config.text_encoder)
            if not os.path.exists(text_encoder_model_path):
                snapshot_download(config.text_encoder, local_dir=text_encoder_model_path, allow_file_pattern="*")

        if config.vae_encoder:
            vae_encoder_model_path = os.path.join(self.model_path, config.vae_encoder)
            if not os.path.exists(vae_encoder_model_path):
                snapshot_download(config.vae_encoder, local_dir=vae_encoder_model_path, allow_file_pattern="*")

        if config.tokenizer:
            tokenizer_path = os.path.join(self.model_path, config.tokenizer)
            if not os.path.exists(tokenizer_path):
                snapshot_download(config.tokenizer, local_dir=tokenizer_path, allow_file_pattern="*")
    
    def generate(
        self,
        positive_magic:str,
        prompt: str,
        edit_image_path: str = "",
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成图像
        
        参数:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            height: 图像高度
            width: 图像宽度
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            seed: 随机种子
            output_path: 输出路径，如果为None则自动生成
            
        返回:
            生成的图像路径
        """
        if self.pipe is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        try:
            logger.info(f"开始生成图像，提示词: {prompt[:50]}...")

            if edit_image_path:
                logger.info(f"使用输入图像进行编辑: {edit_image_path}")
                # 编辑图像
                if self.current_model_id == "Qwen/Qwen-Image-Edit-2511":
                    image = self.pipe(
                        prompt=positive_magic + prompt,
                        # negative_prompt=negative_prompt,
                        edit_image=[Image.open(edit_image_path)],
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        cfg_scale=guidance_scale,
                        seed=seed,
                        edit_image_auto_resize=True,
                        zero_cond_t=True)
                else:
                    image = self.pipe(
                        prompt=positive_magic+prompt,
                        # negative_prompt=negative_prompt,
                        edit_image=[Image.open(edit_image_path)],
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        cfg_scale=guidance_scale,
                        seed=seed,
                        edit_image_auto_resize=True)
            else:
                # 生成图像
                image = self.pipe(
                    prompt=positive_magic+prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    cfg_scale=guidance_scale,
                    seed=seed
                )


            # 保存图像
            if output_path is None:
                raise ValueError("output_path不能为空，请指定输出路径")

            if os.path.isdir(output_path):
                output_dir = Path(output_path).joinpath(Path("outputs/text_to_image"))
                output_dir.mkdir(parents=True, exist_ok=True)
                import uuid
                output_path = str(output_dir / f"{uuid.uuid4()}.png")
            image.save(output_path)
            logger.info(f"图像已保存至: {output_path}")

            return output_path
            
        except Exception as e:
            logger.error(f"生成图像失败: {e}")
            traceback.print_exc()
            raise
    
    def batch_generate(
        self,
        positive_magic:str,
        prompts: list[str],
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> list[str]:
        """
        批量生成图像
        
        参数:
            prompts: 提示词列表
            negative_prompt: 负向提示词
            height: 图像高度
            width: 图像宽度
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            seed: 随机种子
            output_dir: 输出目录
            
        返回:
            生成的图像路径列表
        """
        if output_dir is None:
            output_dir = "outputs/text_to_image"
        
        output_paths = []
        for i, prompt in enumerate(prompts):
            logger.info(f"批量生成 {i+1}/{len(prompts)}")
            import uuid
            output_path = os.path.join(output_dir, f"image_{uuid.uuid4()}_{i:04d}.jpg")
            
            # 如果指定了种子，为每张图像设置不同的种子
            current_seed = seed + i if seed is not None else None
            
            path = self.generate(
                positive_magic=positive_magic,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=current_seed,
                output_path=output_path,
            )
            output_paths.append(path)
        
        return output_paths
    
    def unload_model(self):
        """卸载模型，释放内存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型已卸载")

    def unload_lora(self):
        if self.pipe is not None and self.lora_loaded:
            self.pipe.clear_lora()
            logger.info("lora模型已卸载")


# 全局生成器实例
_generator: Optional[TextToImageGenerator] = None


def get_generator() -> TextToImageGenerator:
    """获取或创建文生图生成器实例"""
    global _generator
    if _generator is None:
        from gradio_app import get_diffsynth_model_dir
        _generator = TextToImageGenerator(get_diffsynth_model_dir())
    return _generator


def generate_image(
    positive_magic:str,
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "MusePublic/Qwen-image",
    lora_model:str=None,
    offload_model:bool=False,
    device: str = "cuda",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    便捷函数：生成图像
    
    参数:
        prompt: 正向提示词
        negative_prompt: 负向提示词
        model_type: 模型类型
        device: 运行设备 (cuda, cpu, mps)
        height: 图像高度
        width: 图像宽度
        num_inference_steps: 推理步数
        guidance_scale: 引导系数
        seed: 随机种子
        output_path: 输出路径
        
    返回:
        生成的图像路径
    """
    generator = get_generator()

    # 更新设备设置
    generator.device = get_gpu_devices()
    # 如果有已加载的模型，需要卸载重新加载
    if generator.pipe is not None:
        generator.unload_model()

    # 如果需要切换模型类型，先卸载旧模型
    if generator.current_model_id is not None and generator.current_model_id != model_id:
        logger.info(f"切换模型: {generator.current_model_id} -> {model_id}")
        generator.unload_model()

    # 如果模型未加载，加载新模型
    if generator.pipe is None:
        generator.load_model(model_id=model_id, lora_model=lora_model, low_varam=offload_model)
    # 如果模型已加载但需要添加/更换LoRA
    elif lora_model and not generator.lora_loaded:
        generator.lora_loaded = False
        generator.unload_lora()
        generator.load_lora_model(lora_model)
        generator.lora_loaded = True

    return generator.generate(
        positive_magic=positive_magic,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        output_path=output_path,
    )

def edit_image(
    input_image_path: str,
    prompt: str,
    negative_prompt: str = "",
    model_id: str = "Qwen-Image-Edit-2511",
    lora_model: str = None,
    offload_model: bool = False,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.5,
    seed: Optional[int] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    图片编辑函数

    参数:
        input_image_path: 输入图片路径
        prompt: 编辑提示词
        negative_prompt: 负向提示词
        model_type: 模型类型
        lora_model: LoRA模型
        offload_model: 是否启用显存管理
        height: 图像高度
        width: 图像宽度
        num_inference_steps: 推理步数
        guidance_scale: 引导系数
        seed: 随机种子
        output_path: 输出路径

    返回:
        编辑后的图像路径
    """

    generator = get_generator()

    # 更新设备设置
    generator.device = get_gpu_devices()
    # 如果有已加载的模型，需要卸载重新加载
    if generator.pipe is not None:
        generator.unload_model()

    # 如果需要切换模型类型，先卸载旧模型
    if generator.current_model_id is not None and generator.current_model_id != model_id:
        logger.info(f"切换模型: {generator.current_model_id} -> {model_id}")
        generator.unload_model()

    # 如果模型未加载，加载新模型
    if generator.pipe is None:
        generator.load_model(PipelineType.IMAGE_EDIT,model_id, lora_model=lora_model, low_varam=offload_model)
    # 如果模型已加载但需要添加/更换LoRA
    elif lora_model and not generator.lora_loaded:
        generator.load_model(PipelineType.IMAGE_EDIT,model_id, lora_model=lora_model, low_varam=offload_model)

    # 使用模型编辑图像
    try:
        logger.info(f"开始编辑图像，提示词: {prompt[:50]}...")

        # 生成图像
        return generator.generate(
            positive_magic="",
            prompt=prompt,
            negative_prompt=negative_prompt,
            edit_image_path=input_image_path,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            output_path=output_path,
        )

    except Exception as e:
        logger.error(f"编辑图像失败: {e}")
        raise

def unload_lora():
    """卸载LoRA模型"""
    generator = get_generator()
    if generator.lora_loaded:
        generator.unload_lora()
        generator.lora_loaded = False

def unload_model():
    """卸载主模型"""
    generator = get_generator()
    generator.unload_model()

def get_model_status():
    """获取当前模型状态"""
    generator = get_generator()
    is_loaded = generator.pipe is not None
    lora_loaded = generator.lora_loaded if is_loaded else False
    return is_loaded, lora_loaded
