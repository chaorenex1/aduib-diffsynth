# Aduib DiffSynth - Copilot Agent Instructions

## Project Overview

**Aduib DiffSynth** is a Python FastAPI service for AI image generation (DiffSynth Engine: SD/SDXL/SD3/Flux/HunyuanDiT), PDF processing (MinerU), and blog management with Gradio UI and MCP integration.

**Tech Stack**: Python 3.12, FastAPI, Gradio, SQLAlchemy 2.0.39, PostgreSQL, Redis, PyTorch 2.7.1 (CUDA 12.8), S3 storage
**Size**: 17MB, 578 Python files, 3,600 LOC

## Critical Build Information

### Package Manager: uv (Required)

**Do NOT use pip install -e .** - Use `uv` package manager only.

Install uv: `pip install uv` (or brew/choco)

### ⚠️ CRITICAL: Aliyun Mirror Connectivity Issue

`pyproject.toml` defaults to `http://mirrors.aliyun.com/pypi/simple/` which often fails with DNS errors outside China.

**ALWAYS use this command for dependencies**:
```bash
rm -f uv.lock  # Remove lock file first
uv sync --dev --index-url https://pypi.org/simple/
source .venv/bin/activate
```

**Installation time**: 2-5 minutes

**Optional extras**: `--extra diffsynth --extra cuda` (image generation), `--extra pdf_parser` (MinerU), `--extra ocr` (PaddleOCR)

## Database Setup (Required)

PostgreSQL required. Configure in `.env` (see `.env.example`). **ALWAYS run in this order**:
```bash
uv sync --dev --index-url https://pypi.org/simple/
alembic -c ./alembic/alembic.ini revision --autogenerate -m "init table"  # Only if no migrations
alembic -c ./alembic/alembic.ini upgrade head
```

## Running Applications

**FastAPI** (port 5002): `python app.py`
**Gradio UI** (port 7860): `python gradio_app.py`

Requires `.env` file (copy from `.env.example`). Set `DEBUG=True` for development.

## Linting: ruff

Config: `.ruff.toml` (line-length 120, excludes `alembic/*`)

```bash
ruff check .          # Check only
ruff check . --fix    # Auto-fix imports and simple issues
```

**Common issues**:
- `I001` Import sorting - safe to auto-fix
- `G004` f-strings in logging - use `log.info("msg %s", var)` not `log.info(f"msg {var}")`
- `E501` Line >120 chars - manually fix

**ALWAYS run `ruff check . --fix` before committing**

## Testing: pytest

Location: `test/` directory (7 tests)

```bash
pytest                              # Run all
pytest test/test_text_to_image.py   # Specific file
pytest --collect-only               # List tests
```

**Known issues**: `test_markdown_example.py` fails without full dependencies; image tests need large model downloads.

## Project Structure

```
aduib-diffsynth/
├── app.py                    # Main FastAPI entry (→ app_factory.py → aduib_app.py)
├── gradio_app.py             # Gradio UI entry
├── alembic/                  # DB migrations (alembic.ini config)
├── configs/                  # Pydantic config (app_config.py main)
├── controllers/              # API endpoints
│   ├── route.py              # Main router registration
│   └── text_to_image.py      # Image generation API
├── diffsynths/               # Core logic
│   ├── text_to_image.py      # TextToImageGenerator class
│   ├── blog.py               # Blog/content generation
│   └── mineru.py             # PDF parsing
├── models/                   # SQLAlchemy ORM models
├── libs/                     # Auth, context, middleware
├── utils/                    # Utilities (encoders, http, markdown)
├── component/                # Cache (Redis), storage (S3), logging
├── test/                     # Test suite
├── pyproject.toml            # Dependencies & config
├── .ruff.toml                # Linter config
└── .env.example              # Environment template
```

**Key patterns**:
- API routes registered in `controllers/route.py`
- Config uses Pydantic Settings + Nacos remote source
- Middleware: auth (ApiKeyContextMiddleware), logging, tracing

## Common Tasks

**Add API endpoint**: Create `controllers/your_feature.py` with FastAPI router → register in `controllers/route.py`
**Add DB model**: Create in `models/`, run `alembic -c ./alembic/alembic.ini revision --autogenerate -m "msg"`, then `upgrade head`
**Modify dependencies**: Edit `pyproject.toml` → delete `uv.lock` → re-run `uv sync --dev --index-url https://pypi.org/simple/`

## Important Coding Patterns

**Logging** (ruff G004): Use `log.info("msg %s", var)` NOT `log.info(f"msg {var}")`

**Memory management** (image generation):
```python
from diffsynths.text_to_image import get_generator
generator = get_generator()
generator.load_model("sd")
image_path = generator.generate(prompt="...")
generator.unload_model()  # CRITICAL: Free VRAM
```

**Low VRAM mode**: When `low_vram=True`, apply `vram_config` to ALL ModelConfig instances (including tokenizer_config)

**API auth**: If `AUTH_ENABLED=True`, requests need `X-API-Key` header

## Troubleshooting

**uv sync DNS error**: Remove `uv.lock`, use `--index-url https://pypi.org/simple/` (see above)
**"No module named 'pydantic'"**: Run `uv sync --dev --index-url https://pypi.org/simple/` and activate `.venv`
**"No module named 'configs'"**: Must run from repo root `/home/runner/work/aduib-diffsynth/aduib-diffsynth/`
**CUDA out of memory**: Reduce image size (512x512), fewer steps (20), use `low_vram=True`, or `device="cpu"`
**Port in use**: `lsof -i :5002` then `kill -9 <PID>` or change `APP_PORT` in `.env`
**Alembic fails**: Verify DB connection in `.env`, ensure PostgreSQL running, check DB exists

## Performance Notes

- App init: ~1-2 seconds
- Model download: GB-scale first time, faster subsequent
- Image generation: 5-30 seconds (model/size/steps/hardware dependent)

## Resources

- **README.md**: Quick start
- **docs/TEXT_TO_IMAGE.md**: Comprehensive image generation guide
- **docs/IMAGE_EDITING_API.md**: Image editing documentation
- **.env.example**: All environment variables

## Agent Guidelines

**Trust these instructions** - Only search if incomplete or incorrect for your task.

**Always start**:
1. `rm -f uv.lock && uv sync --dev --index-url https://pypi.org/simple/`
2. `source .venv/bin/activate`
3. Work from repo root: `/home/runner/work/aduib-diffsynth/aduib-diffsynth/`

**Before committing**:
- Run `ruff check . --fix`
- Manually fix remaining errors (line length, etc.)
- Test changes manually (limited test coverage)

**No CI configured** - Manual validation only.
