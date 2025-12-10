"""Minimal Gradio UI to demo DiffSynth functionality."""

import gradio as gr

from app_factory import create_app
from diffsynths.blog import process_pdf_files, upload_to_blog, create_blog
from diffsynths.text_to_image import generate_image, unload_model, unload_lora, get_model_status, edit_image

app=create_app()
mineru_working_dir = app.app_home + "/mineru"
diffsynth_working_dir = app.app_home + "/diffsynth"
diffsynth_model_dir = app.app_home + "/diffsynth/model"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="DiffSynth Playground") as gradio_app:
        gr.Markdown("# DiffSynth Playground\n")
        # mineru tab页
        with gr.Tab("MINERU"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 1. 选择文件或者目录
                    files = gr.Files(label="Upload PDF Files", file_types=[".pdf"])
                    # 2. 选择语言
                    lang = gr.Dropdown(
                        choices=["auto", "en", "zh"],
                        value="zh",
                        label="Select Language",
                    )
                    # 3. 选择方式
                    method = gr.Dropdown(
                        choices=["auto", "txt", "ocr"],
                        value="ocr",
                        label="Select Parsing Method",
                    )
                    submit_button = gr.Button("Submit")
                with gr.Column(scale=1):
                    # 4. 显示结果
                    output_box = gr.HTML(label="Output", value="<div>等待上传文件...</div>")
                    # 5. 下载 ZIP 文件
                    download_file = gr.File(label="Download Results (ZIP)")
                    # 6. 上传blog
                    upload_file_button = gr.Button("Upload to Blog")
                    upload_file_button.click(
                        fn=upload_to_blog,
                        inputs=[download_file],
                        outputs=[output_box]
                    )
            submit_button.click(
                fn=process_pdf_files,
                inputs=[files, lang, method],
                outputs=[output_box, download_file],
            )
        with gr.Tab("Aduib Blog RAG"):
            with gr.Column(scale=1):
                # 1. 选择 Markdown 文件
                md_files = gr.Files(label="Upload Markdown File", file_types=[".md"])
                upload_md_button = gr.Button("Upload Markdown to Blog")
                md_output_box = gr.HTML(label="Output", value="<div>等待上传文件...</div>")
            upload_md_button.click(
                fn=create_blog,
                inputs=[md_files],
                outputs=[md_output_box],
            )
        with gr.Tab("文生图"):
            gr.Markdown("## Text-to-Image Generation\n使用 DiffSynth Engine 生成图像")

            with gr.Row():
                with gr.Column(scale=1):
                    # 模型状态显示和卸载按钮
                    with gr.Row():
                        model_status_text = gr.Textbox(
                            label="模型状态",
                            value="未加载",
                            interactive=False,
                            scale=2,
                        )
                        lora_status_text = gr.Textbox(
                            label="LoRA状态",
                            value="未加载",
                            interactive=False,
                            scale=2,
                        )

                    with gr.Row():
                        unload_model_button = gr.Button("🗑️ 卸载模型", variant="secondary")
                        unload_lora_button = gr.Button("🗑️ 卸载LoRA", variant="secondary")
                        refresh_status_button = gr.Button("🔄 刷新状态", variant="secondary")

                    # positive_magic
                    positive_magic_input=gr.Textbox(label="Positive Magic",
                                              placeholder="在提示词前添加以增强效果",
                                              lines=2,
                                              value="masterpiece, best quality, ultra-detailed, 8k, high resolution, cinematic lighting, intricate details, photorealistic, sharp focus, vibrant colors")
                    # 提示词输入
                    prompt_input = gr.Textbox(
                        label="正向提示词 (Prompt)",
                        placeholder="输入您想生成的图像描述...",
                        lines=4,
                    )
                    negative_prompt_input = gr.Textbox(
                        label="负向提示词 (Negative Prompt)",
                        placeholder="输入不想出现的元素...",
                        lines=4,
                        value="网格化，规则的网格，模糊, 低分辨率, 低质量, 变形, 畸形, 错误的解剖学, 变形的手, 变形的身体, 变形的脸, 变形的头发, 变形的眼睛, 变形的嘴巴",
                    )

                    # 模型选择
                    model_type_dropdown = gr.Dropdown(
                        choices=["Qwen-Image","Qwen-Image-Edit","MusePublic/Qwen-image","Qwen Image FP8"],
                        value="MusePublic/Qwen-image",
                        label="模型类型",
                        info="选择不同的扩散模型",
                    )

                    # lora选择（可选）
                    lora_dropdown = gr.Dropdown(
                        choices=[
                            ("不使用 LoRA", "none"),
                            ("肉色连裤袜 LoRA", "animationtj/Qwen_image_nude_pantyhose_lora"),
                            ("麦橘千问美人 LoRA", "merjic/majicbeauty-qwen1"),
                            ("美学提升 LoRA","DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1"),
                            ("多语言对齐 LoRA","DiffSynth-Studio/Qwen-Image-LangAlign-LoRA")
                        ],
                        value="none",
                        label="LoRA 模型 (可选)",
                        info="选择 LoRA 模型以微调生成效果",
                    )

                    # offload
                    offload_checkbox = gr.Checkbox(
                        label="启用模型卸载 (Offload)",
                        value=False,
                        info="启用后可在低显存设备上运行，但速度较慢",
                    )

                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="宽度",
                        )
                        height_slider = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="高度",
                        )

                    with gr.Row():
                        steps_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=40,
                            step=1,
                            label="推理步数",
                        )
                        guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=4.0,
                            step=0.5,
                            label="引导系数 (CFG Scale)",
                        )

                    seed_input = gr.Number(
                        label="随机种子 (Seed)",
                        value=42,
                        precision=0,
                        info="设置为 40 使用随机种子",
                    )

                    generate_button = gr.Button("🎨 生成图像", variant="primary")

                with gr.Column(scale=1):
                    # 输出图像
                    output_image = gr.Image(
                        label="生成的图像",
                        type="filepath",
                    )
                    output_info = gr.Textbox(
                        label="生成信息",
                        lines=2,
                    )

            # 定义生成函数
            def generate_image_gradio(
                positive_magic,prompt, negative_prompt, model_type,lora_model,offload, width, height, steps, guidance, seed
            ):
                try:
                    import time
                    start_time = time.time()

                    # 处理种子值
                    seed_value = None if seed == -1 else int(seed)

                    # 生成图像
                    image_path = generate_image(
                        positive_magic=positive_magic,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model_type=model_type,
                        lora_model=lora_model if lora_model != "none" else None,
                        offload_model=offload,
                        width=int(width),
                        height=int(height),
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        seed=seed_value,
                        output_path=diffsynth_working_dir,
                    )

                    elapsed_time = time.time() - start_time
                    info = f"✅ 生成成功！\n耗时: {elapsed_time:.2f}秒\n图像路径: {image_path}"

                    # 更新状态
                    model_loaded, lora_loaded = get_model_status()
                    model_status = f"✅ 已加载 ({model_type})" if model_loaded else "❌ 未加载"
                    lora_status = f"✅ 已加载 ({lora_model})" if lora_loaded else "❌ 未加载"

                    return image_path, info, model_status, lora_status

                except Exception as e:
                    error_info = f"❌ 生成失败: {str(e)}"
                    # 获取当前状态
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "✅ 已加载" if lora_loaded else "❌ 未加载"
                    return None, error_info, model_status, lora_status

            # 定义卸载模型函数
            def unload_model_gradio():
                try:
                    unload_model()
                    return "❌ 未加载", "❌ 未加载", "✅ 模型已成功卸载"
                except Exception as e:
                    return "⚠️ 状态未知", "⚠️ 状态未知", f"❌ 卸载失败: {str(e)}"

            # 定义卸载LoRA函数
            def unload_lora_gradio():
                try:
                    unload_lora()
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "❌ 未加载"
                    return model_status, lora_status, "✅ LoRA已成功卸载"
                except Exception as e:
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "✅ 已加载" if lora_loaded else "❌ 未加载"
                    return model_status, lora_status, f"❌ 卸载失败: {str(e)}"

            # 定义刷新状态函数
            def refresh_status_gradio():
                try:
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "✅ 已加载" if lora_loaded else "❌ 未加载"
                    return model_status, lora_status
                except Exception as e:
                    return "⚠️ 状态未知", "⚠️ 状态未知"

            # 绑定事件
            generate_button.click(
                fn=generate_image_gradio,
                inputs=[
                    positive_magic_input,
                    prompt_input,
                    negative_prompt_input,
                    model_type_dropdown,
                    lora_dropdown,
                    offload_checkbox,
                    width_slider,
                    height_slider,
                    steps_slider,
                    guidance_slider,
                    seed_input,
                ],
                outputs=[output_image, output_info, model_status_text, lora_status_text],
            )

            # 绑定卸载模型按钮
            unload_model_button.click(
                fn=unload_model_gradio,
                inputs=[],
                outputs=[model_status_text, lora_status_text, output_info],
            )

            # 绑定卸载LoRA按钮
            unload_lora_button.click(
                fn=unload_lora_gradio,
                inputs=[],
                outputs=[model_status_text, lora_status_text, output_info],
            )

            # 绑定刷新状态按钮
            refresh_status_button.click(
                fn=refresh_status_gradio,
                inputs=[],
                outputs=[model_status_text, lora_status_text],
            )

        with gr.Tab("图片编辑"):
            gr.Markdown("## Image Editing\n使用 Qwen-Image-Edit 模型编辑图像")

            with gr.Row():
                with gr.Column(scale=1):
                    # 模型状态显示
                    with gr.Row():
                        edit_model_status_text = gr.Textbox(
                            label="模型状态",
                            value="未加载",
                            interactive=False,
                            scale=2,
                        )
                        edit_lora_status_text = gr.Textbox(
                            label="LoRA状态",
                            value="未加载",
                            interactive=False,
                            scale=2,
                        )

                    with gr.Row():
                        edit_unload_model_button = gr.Button("🗑️ 卸载模型", variant="secondary")
                        edit_unload_lora_button = gr.Button("🗑️ 卸载LoRA", variant="secondary")
                        edit_refresh_status_button = gr.Button("🔄 刷新状态", variant="secondary")

                    # 上传输入图片
                    input_image = gr.Image(
                        label="上传原始图片",
                        type="filepath",
                        sources=["upload", "clipboard"],
                    )

                    # 提示词输入
                    edit_prompt_input = gr.Textbox(
                        label="编辑提示词 (Prompt)",
                        placeholder="描述您想要的编辑效果...",
                        lines=4,
                    )
                    edit_negative_prompt_input = gr.Textbox(
                        label="负向提示词 (Negative Prompt)",
                        placeholder="输入不想出现的元素...",
                        lines=4,
                        value="模糊, 低分辨率, 低质量, 变形, 畸形, 错误的解剖学",
                    )

                    # 模型选择
                    edit_model_type_dropdown = gr.Dropdown(
                        choices=["Qwen-Image-Edit-2509"],
                        value="Qwen-Image-Edit-2509",
                        label="模型类型",
                        info="选择图片编辑模型",
                    )

                    # lora选择（可选）
                    edit_lora_dropdown = gr.Dropdown(
                        choices=[
                            ("不使用 LoRA", "none"),
                            ("美学提升 LoRA","DiffSynth-Studio/Qwen-Image-LoRA-ArtAug-v1"),
                        ],
                        value="none",
                        label="LoRA 模型 (可选)",
                        info="选择 LoRA 模型以微调编辑效果",
                    )

                    # offload
                    edit_offload_checkbox = gr.Checkbox(
                        label="启用模型卸载 (Offload)",
                        value=False,
                        info="启用后可在低显存设备上运行，但速度较慢",
                    )

                    with gr.Row():
                        edit_width_slider = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="宽度",
                        )
                        edit_height_slider = gr.Slider(
                            minimum=256,
                            maximum=2048,
                            value=1024,
                            step=64,
                            label="高度",
                        )

                    with gr.Row():
                        edit_steps_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="推理步数",
                        )
                        edit_guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=4.5,
                            step=0.5,
                            label="引导系数 (CFG Scale)",
                        )

                    edit_seed_input = gr.Number(
                        label="随机种子 (Seed)",
                        value=42,
                        precision=0,
                        info="设置为 -1 使用随机种子",
                    )

                    edit_generate_button = gr.Button("✨ 编辑图像", variant="primary")

                with gr.Column(scale=1):
                    # 输出图像
                    edit_output_image = gr.Image(
                        label="编辑后的图像",
                        type="filepath",
                    )
                    edit_output_info = gr.Textbox(
                        label="生成信息",
                        lines=2,
                    )

            # 定义编辑函数
            def edit_image_gradio(
                input_img, prompt, negative_prompt, model_type, lora_model, offload,
                width, height, steps, guidance, seed
            ):
                try:
                    if input_img is None:
                        return None, "❌ 请先上传图片", "⚠️ 状态未知", "⚠️ 状态未知"

                    import time
                    start_time = time.time()

                    # 处理种子值
                    seed_value = None if seed == -1 else int(seed)

                    # 编辑图像
                    image_path = edit_image(
                        input_image_path=input_img,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model_type=model_type,
                        lora_model=lora_model if lora_model != "none" else None,
                        offload_model=offload,
                        width=int(width),
                        height=int(height),
                        num_inference_steps=int(steps),
                        guidance_scale=guidance,
                        seed=seed_value,
                        output_path=diffsynth_working_dir,
                    )

                    elapsed_time = time.time() - start_time
                    info = f"✅ 编辑成功！\n耗时: {elapsed_time:.2f}秒\n图像路径: {image_path}"

                    # 更新状态
                    model_loaded, lora_loaded = get_model_status()
                    model_status = f"✅ 已加载 ({model_type})" if model_loaded else "❌ 未加载"
                    lora_status = f"✅ 已加载 ({lora_model})" if lora_loaded else "❌ 未加载"

                    return image_path, info, model_status, lora_status

                except Exception as e:
                    error_info = f"❌ 编辑失败: {str(e)}"
                    # 获取当前状态
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "✅ 已加载" if lora_loaded else "❌ 未加载"
                    return None, error_info, model_status, lora_status

            # 定义卸载模型函数 (图片编辑)
            def edit_unload_model_gradio():
                try:
                    unload_model()
                    return "❌ 未加载", "❌ 未加载", "✅ 模型已成功卸载"
                except Exception as e:
                    return "⚠️ 状态未知", "⚠️ 状态未知", f"❌ 卸载失败: {str(e)}"

            # 定义卸载LoRA函数 (图片编辑)
            def edit_unload_lora_gradio():
                try:
                    unload_lora()
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "❌ 未加载"
                    return model_status, lora_status, "✅ LoRA已成功卸载"
                except Exception as e:
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "✅ 已加载" if lora_loaded else "❌ 未加载"
                    return model_status, lora_status, f"❌ 卸载失败: {str(e)}"

            # 定义刷新状态函数 (图片编辑)
            def edit_refresh_status_gradio():
                try:
                    model_loaded, lora_loaded = get_model_status()
                    model_status = "✅ 已加载" if model_loaded else "❌ 未加载"
                    lora_status = "✅ 已加载" if lora_loaded else "❌ 未加载"
                    return model_status, lora_status
                except Exception as e:
                    return "⚠️ 状态未知", "⚠️ 状态未知"

            # 绑定事件
            edit_generate_button.click(
                fn=edit_image_gradio,
                inputs=[
                    input_image,
                    edit_prompt_input,
                    edit_negative_prompt_input,
                    edit_model_type_dropdown,
                    edit_lora_dropdown,
                    edit_offload_checkbox,
                    edit_width_slider,
                    edit_height_slider,
                    edit_steps_slider,
                    edit_guidance_slider,
                    edit_seed_input,
                ],
                outputs=[edit_output_image, edit_output_info, edit_model_status_text, edit_lora_status_text],
            )

            # 绑定卸载模型按钮
            edit_unload_model_button.click(
                fn=edit_unload_model_gradio,
                inputs=[],
                outputs=[edit_model_status_text, edit_lora_status_text, edit_output_info],
            )

            # 绑定卸载LoRA按钮
            edit_unload_lora_button.click(
                fn=edit_unload_lora_gradio,
                inputs=[],
                outputs=[edit_model_status_text, edit_lora_status_text, edit_output_info],
            )

            # 绑定刷新状态按钮
            edit_refresh_status_button.click(
                fn=edit_refresh_status_gradio,
                inputs=[],
                outputs=[edit_model_status_text, edit_lora_status_text],
            )

    return gradio_app


def main():
    gradio_app = build_interface()
    gradio_app.launch(server_name="0.0.0.0", server_port=7860, show_error=True,mcp_server=True,allowed_paths=[app.app_home])


if __name__ == "__main__":
    main()