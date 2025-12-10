"""
Text-to-Image generation using DiffSynth Engine
"""
import logging
import os
from pathlib import Path
from typing import Optional, Literal

import torch
from modelscope import snapshot_download

logger = logging.getLogger(__name__)

vram_config = {
    "offload_dtype": torch.float8_e4m3fn,
    "offload_device": "cpu",
    # "onload_dtype": torch.float8_e4m3fn,
    # "onload_device": "cpu",
    # "preparing_dtype": torch.float8_e4m3fn,
    # "preparing_device": "cuda",
    # "computation_dtype": torch.bfloat16,
    # "computation_device": "cuda",
}


class TextToImageGenerator:
    """文生图生成器，使用 DiffSynth Engine"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化文生图生成器
        
        参数:
            model_path: 模型路径，如果为None则使用默认模型
            device: 运行设备，cuda或cpu
        """
        self.current_model_type = None
        self.device = device
        self.model_path = model_path
        self.pipe = None
        self.lora_loaded=False
        if not os.path.exists(model_path):
            os.makedirs(model_path,exist_ok=True)
        logger.info(f"初始化 TextToImageGenerator，设备: {device}")
        
    def load_model(
        self,
        model_type: str = "MusePublic/Qwen-image",
        lora_model:str=None,
        low_varam: bool = False,
    ):
        """
        加载模型
        
        参数:
            model_type: 模型类型，支持 sd, sdxl, sd3, flux, hunyuan
        """
        try:
            self.current_model_type = model_type
            # 根据模型类型选择对应的 Pipeline
            if model_type == "sd":
                logger.info("加载 Stable Diffusion 模型...")

            elif model_type == "sdxl":
                logger.info("加载 Stable Diffusion XL 模型...")
                
            elif model_type == "sd3":
                logger.info("加载 Stable Diffusion 3 模型...")
                
            elif model_type == "flux":
                logger.info("加载 Flux 模型...")
                
            elif model_type == "hunyuan":
                logger.info("加载 HunyuanDiT 模型...")
            elif model_type == "MusePublic/Qwen-image":
                logger.info("加载 MusePublic/Qwen-image 模型...")
                from diffsynth.pipelines.qwen_image import QwenImagePipeline,ModelConfig
                if not os.path.exists(os.path.join(self.model_path,"MusePublic/Qwen-image")):
                    snapshot_download("MusePublic/Qwen-image", local_dir=self.model_path+"/MusePublic/Qwen-image", allow_file_pattern="*")
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device=self.device,
                    model_configs=[
                        ModelConfig(model_id="MusePublic/Qwen-image",
                                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="MusePublic/Qwen-image", origin_file_pattern="text_encoder/model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="MusePublic/Qwen-image",
                                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",skip_download=True,local_model_path=self.model_path),
                    ],
                    tokenizer_config=ModelConfig(model_id="MusePublic/Qwen-image", origin_file_pattern="tokenizer/",skip_download=True,local_model_path=self.model_path),
                )
                if lora_model:
                    self.pipe.load_lora(self.pipe.dit, lora_config=ModelConfig(model_id=lora_model, origin_file_pattern="model.safetensors"))
                    self.lora_loaded=True
            elif model_type == "Qwen Image FP8":
                logger.info("加载 Qwen Image FP8 模型...")
                from diffsynth.pipelines.qwen_image import QwenImagePipeline,ModelConfig
                #判断模型是否下载
                if not os.path.exists(os.path.join(self.model_path,"MusePublic/Qwen-image-fp8")):
                    snapshot_download("MusePublic/Qwen-image-fp8", local_dir=self.model_path+"/MusePublic/Qwen-image-fp8", allow_file_pattern="*")
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=torch.float8_e4m3fn,
                    device=self.device,
                    model_configs=[
                        ModelConfig(model_id="MusePublic/Qwen-image-fp8",
                                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="MusePublic/Qwen-image-fp8", origin_file_pattern="text_encoder/model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="MusePublic/Qwen-image-fp8",
                                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",skip_download=True,local_model_path=self.model_path),
                    ],
                    tokenizer_config=ModelConfig(model_id="MusePublic/Qwen-image-fp8", origin_file_pattern="tokenizer/",skip_download=True,local_model_path=self.model_path),
                )
                if lora_model:
                    self.pipe.load_lora(self.pipe.dit, lora_config=ModelConfig(model_id=lora_model, origin_file_pattern="model.safetensors"))
                    self.lora_loaded=True
            elif model_type == "Qwen-Image":
                logger.info("加载 Qwen-Image 模型...")
                from diffsynth.pipelines.qwen_image import QwenImagePipeline,ModelConfig
                if not os.path.exists(os.path.join(self.model_path,"Qwen/Qwen-Image")):
                    snapshot_download("Qwen/Qwen-Image", local_dir=self.model_path+"/Qwen/Qwen-Image", allow_file_pattern="*")
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device=self.device,
                    model_configs=[
                        ModelConfig(model_id="Qwen/Qwen-Image",
                                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="Qwen/Qwen-Image",
                                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",skip_download=True,local_model_path=self.model_path),
                    ],
                    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
                )
                if lora_model:
                    self.pipe.load_lora(self.pipe.dit, lora_config=ModelConfig(model_id=lora_model, origin_file_pattern="model.safetensors"))
                    self.lora_loaded=True
            elif model_type == "Qwen-Image-Edit-2509":
                logger.info("加载 Qwen-Image-Edit-2509 模型...")
                from diffsynth.pipelines.qwen_image import QwenImagePipeline,ModelConfig
                if not os.path.exists(os.path.join(self.model_path,"Qwen/Qwen-Image-Edit-2509")):
                    snapshot_download("Qwen/Qwen-Image-Edit-2509", local_dir=self.model_path+"/Qwen/Qwen-Image-Edit-2509", allow_file_pattern="*")
                self.pipe = QwenImagePipeline.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    device=self.device,
                    model_configs=[
                        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509",
                                    origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="Qwen/Qwen-Image",
                                    origin_file_pattern="text_encoder/model*.safetensors",skip_download=True,local_model_path=self.model_path),
                        ModelConfig(model_id="Qwen/Qwen-Image",
                                    origin_file_pattern="vae/diffusion_pytorch_model.safetensors",skip_download=True,local_model_path=self.model_path),
                    ],
                    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
                )
                if lora_model:
                    lora_config = ModelConfig(model_id=lora_model, origin_file_pattern="model.safetensors",
                                         skip_download=True, local_model_path=self.model_path)
                    if not os.path.exists(os.path.join(self.model_path,lora_model,"model.safetensors")):
                        snapshot_download(lora_model, local_dir=os.path.join(self.model_path,lora_model), allow_file_pattern="model.safetensors*")
                    self.pipe.load_lora(self.pipe.dit, lora_config=lora_config)
                    self.lora_loaded = True
                if low_varam:
                    self.pipe.enable_vram_management()
            
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
            logger.info(f"模型加载完成: {model_type}")
            
        except ImportError as e:
            logger.error(f"导入 diffsynth 模块失败: {e}")
            logger.error("请确保已安装 diffsynth-engine: pip install diffsynth-engine")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def generate(
        self,
        positive_magic:str,
        prompt: str,
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
            
            # 生成图像
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                cfg_scale=guidance_scale,
                seed=seed
            )


            # 保存图像
            if output_path is None:
                output_dir = Path("outputs/text_to_image")
                output_dir.mkdir(parents=True, exist_ok=True)
                import uuid
                output_path = str(output_dir / f"{uuid.uuid4()}.png")
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            image.save(output_path)
            logger.info(f"图像已保存至: {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"生成图像失败: {e}")
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
            output_path = os.path.join(output_dir, f"image_{i:04d}.png")
            
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
        from gradio_app import diffsynth_model_dir
        _generator = TextToImageGenerator(diffsynth_model_dir)
    return _generator


def generate_image(
    positive_magic:str,
    prompt: str,
    negative_prompt: str = "",
    model_type: str = "MusePublic/Qwen-image",
    lora_model:str=None,
    offload_model:bool=False,
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
    
    # 如果需要切换模型类型，先卸载旧模型
    if generator.current_model_type is not None and generator.current_model_type != model_type:
        logger.info(f"切换模型: {generator.current_model_type} -> {model_type}")
        generator.unload_model()

    # 如果模型未加载，加载新模型
    if generator.pipe is None:
        generator.load_model(model_type, lora_model=lora_model, low_varam=offload_model)
    # 如果模型已加载但需要添加/更换LoRA
    elif lora_model and not generator.lora_loaded:
        generator.load_model(model_type, lora_model=lora_model, low_varam=offload_model)

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


def edit_image(
    input_image_path: str,
    prompt: str,
    negative_prompt: str = "",
    model_type: str = "Qwen-Image-Edit-2509",
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
    from PIL import Image

    generator = get_generator()

    # 如果需要切换模型类型，先卸载旧模型
    if generator.current_model_type is not None and generator.current_model_type != model_type:
        logger.info(f"切换模型: {generator.current_model_type} -> {model_type}")
        generator.unload_model()

    # 如果模型未加载，加载新模型
    if generator.pipe is None:
        generator.load_model(model_type, lora_model=lora_model, low_varam=offload_model)
    # 如果模型已加载但需要添加/更换LoRA
    elif lora_model and not generator.lora_loaded:
        generator.load_model(model_type, lora_model=lora_model, low_varam=offload_model)

    # 加载输入图像
    input_image = Image.open(input_image_path).convert("RGB")

    # 调整图像尺寸
    if input_image.size != (width, height):
        input_image = input_image.resize((width, height), Image.LANCZOS)

    # 使用模型编辑图像
    try:
        logger.info(f"开始编辑图像，提示词: {prompt[:50]}...")

        # 生成图像
        edited_image = generator.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=input_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg_scale=guidance_scale,
            seed=seed
        )

        # 保存图像
        if output_path is None:
            output_dir = Path("outputs/image_edit")
            output_dir.mkdir(parents=True, exist_ok=True)
            import uuid
            output_path = str(output_dir / f"{uuid.uuid4()}.png")
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        edited_image.save(output_path)
        logger.info(f"编辑后图像已保存至: {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"编辑图像失败: {e}")
        raise


