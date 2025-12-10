# Aduib Diffsynth
## 项目简介

Aduib DiffSynth Service

## 快速开始
1. 安装环境
    ```bash
    pip install uv
    # Or on macOS
    brew install uv
    # Or on Windows
    choco install uv
    ```
2. 安装依赖
   ```bash
   uv sync --dev
    ```
  
3. 初始化数据库
   ```bash
    uv pip install alembic
    alembic -c ./alembic/alembic.ini revision --autogenerate -m "init table"
    alembic -c ./alembic/alembic.ini upgrade head
   ```

## Gradio Demo

1. 安装依赖
   ```bash
   uv sync
   ```
2. 启动 Gradio
   ```bash
   uv run python gradio_app.py
   ```

## 文档 (Docs)

- [docs](docs/README.md) — 文档
