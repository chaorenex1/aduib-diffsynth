# Aduib DiffSynth - Copilot Agent Instructions

## Project Overview

**Aduib DiffSynth** is a Python-based AI image generation and document processing service that provides:
- Text-to-image generation using DiffSynth Engine (supports SD, SDXL, SD3, Flux, HunyuanDiT models)
- Image editing capabilities
- PDF parsing and processing with MinerU
- Blog content creation and management
- FastAPI REST API backend
- Gradio web UI interface
- MCP (Model Context Protocol) integration with Nacos service discovery

**Repository Size**: ~17MB (578 Python files, ~3,600 lines of code)

**Languages/Frameworks**:
- Python 3.11-3.12 (3.12 recommended per `.python-version`)
- FastAPI for REST API
- Gradio for web UI
- SQLAlchemy 2.0.39 for ORM
- PostgreSQL database
- Redis for caching
- PyTorch 2.7.1 (with optional CUDA 12.8 support)
- S3-compatible object storage (via opendal/boto3)

**Key Dependencies**:
- `diffsynth` - Core image generation engine (optional)
- `fastapi>=0.116.1` - Web framework
- `gradio>=5.0.0` - UI framework
- `pydantic>=2.11.3,<=2.12.4` - Data validation (version range is strict)
- `sqlalchemy==2.0.39` - Database ORM (exact version)
- `alembic==1.16.4` - Database migrations
- `pytest==8.4.1` - Testing framework
- `ruff` - Linting and formatting

## Critical Build Information

### Package Manager: uv (Required)

This project **requires** the `uv` package manager. Standard pip/pip3 commands will NOT work properly due to the package structure and pyproject.toml configuration.

**Install uv first**:
```bash
pip install uv
# Or on macOS: brew install uv
# Or on Windows: choco install uv
```

### Known Build Issue: Aliyun Mirror Connectivity

⚠️ **CRITICAL**: The `pyproject.toml` configures Aliyun mirror (`http://mirrors.aliyun.com/pypi/simple/`) as the default package index. This mirror may be inaccessible from certain networks/regions, causing `uv sync` to fail with DNS errors.

**Symptoms**:
```
error: Failed to fetch: `http://mirrors.aliyun.com/pypi/simple/...`
Caused by: failed to lookup address information: No address associated with hostname
```

**Workarounds** (try in this order):

1. **Remove uv.lock and use PyPI override** (most reliable):
   ```bash
   rm -f uv.lock
   uv sync --dev --index-url https://pypi.org/simple/
   ```

2. **Skip Aliyun mirror entirely**: Temporarily comment out the Aliyun index in `pyproject.toml`:
   ```toml
   # [[tool.uv.index]]
   # url = "http://mirrors.aliyun.com/pypi/simple/"
   # default = true
   ```

3. **If in China/Asia**: The mirror might work - try standard `uv sync --dev`

**Do NOT use `pip install -e .`** - This project's setup will fail with "package discovery" errors.

### Standard Build/Install Process

**For development (with test dependencies)**:
```bash
# ALWAYS remove lock file first if experiencing mirror issues
rm -f uv.lock

# Install all dependencies including dev tools (pytest, ruff)
uv sync --dev --index-url https://pypi.org/simple/

# Activate the virtual environment created by uv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**For production (minimal dependencies)**:
```bash
uv sync --index-url https://pypi.org/simple/
```

**Optional extras** (require additional setup/models):
```bash
# Image generation with DiffSynth + CUDA support
uv sync --extra diffsynth --extra cuda --index-url https://pypi.org/simple/

# PDF parsing with MinerU
uv sync --extra pdf_parser --index-url https://pypi.org/simple/

# OCR capabilities with PaddleOCR
uv sync --extra ocr --index-url https://pypi.org/simple/
```

**Installation takes**: ~2-5 minutes depending on extras and network speed.

## Database Setup

This project uses PostgreSQL with Alembic migrations. Database setup is **required** before running the application.

**Prerequisites**:
- PostgreSQL server running and accessible
- Database credentials configured in `.env` file (see `.env.example`)

**Setup steps** (ALWAYS run in this order):
```bash
# 1. Ensure dependencies are installed
uv sync --dev --index-url https://pypi.org/simple/

# 2. Create initial migration (only if no migrations exist in alembic/versions/)
alembic -c ./alembic/alembic.ini revision --autogenerate -m "init table"

# 3. Apply migrations to database
alembic -c ./alembic/alembic.ini upgrade head
```

**Database configuration** (in `.env`):
```env
DB_ENABLED=True
DB_DRIVER=postgresql
DB_HOST=<your-host>
DB_PORT=5432
DB_USERNAME=<username>
DB_PASSWORD=<password>
DB_DATABASE=aduib_ai
```

## Running the Application

### 1. FastAPI Backend

```bash
# Activate venv if not already active
source .venv/bin/activate

# Run the main application
python app.py
```

**Default**: Runs on `http://0.0.0.0:5002` (configurable via `APP_PORT` in `.env`)

**Note**: First run will be slower due to logging initialization and middleware setup.

### 2. Gradio Web UI

```bash
# Activate venv
source .venv/bin/activate

# Run Gradio app
python gradio_app.py
```

**Default**: Gradio runs on `http://localhost:7860`

**Features**: Text-to-image generation, PDF processing (MINERU), blog management

### Environment Configuration

Copy `.env.example` to `.env` and configure:
- `APP_NAME`, `APP_VERSION`, `APP_HOST`, `APP_PORT` - Application settings
- `DEBUG=True` for development (enables verbose logging)
- `AUTH_ENABLED=True` to require API key authentication
- Database settings (see above)
- Redis settings (optional, for caching)
- S3/storage settings (for image uploads)
- Nacos settings (for service discovery, optional)

## Linting and Code Quality

### Linter: ruff

**Configuration**: `.ruff.toml` (line length: 120, excludes `alembic/*`)

**Check code** (read-only):
```bash
ruff check .
```

**Auto-fix issues**:
```bash
ruff check . --fix
```

**Common issues**:
- `I001`: Import sorting - safe to auto-fix
- `E501`: Line too long (>120 chars) - manually fix
- `F821`: Undefined name - check imports
- `G004`: f-strings in logging - use lazy `%` formatting instead

**Ignored rules**: See `.ruff.toml` lines 48-80 for intentionally ignored rules (e.g., `F841` unused-variable, `E722` bare-except)

**Per-file ignores**:
- `__init__.py` files: Allow unused imports (`F401`)
- `configs/*`: Allow invalid function names (`N802`)
- `tests/*`: Allow redefined-while-unused (`F811`)

## Testing

### Test Framework: pytest

**Test location**: `test/` directory

**Available tests**:
- `test/test_text_to_image.py` - Text-to-image generation tests
- `test/test_image_editing_api.py` - Image editing API tests
- `test/test_markdown_example.py` - Markdown processing tests

**Run all tests**:
```bash
pytest
```

**Run specific test file**:
```bash
pytest test/test_text_to_image.py
```

**Collect tests without running**:
```bash
pytest --collect-only
```

**Known test issues**:
- `test_markdown_example.py` may fail on import if dependencies not fully installed
- Image generation tests require DiffSynth models (large downloads ~GB scale)
- Tests that use `torch` require CUDA setup if running on GPU

**Test configuration**: Defined in `pyproject.toml` under `[tool.pytest]` (if present)

## Project Architecture

### Directory Structure

```
aduib-diffsynth/
├── .github/                      # GitHub configuration
│   └── copilot-instructions.md   # This file
├── alembic/                      # Database migrations
│   ├── alembic.ini               # Alembic config
│   ├── env.py                    # Migration environment
│   └── versions/                 # Migration files (auto-generated)
├── component/                    # Application components
│   ├── cache/                    # Redis cache implementation
│   ├── log/                      # Logging setup
│   └── storage/                  # S3/storage implementations
├── configs/                      # Configuration modules
│   ├── app_config.py             # Main app configuration
│   ├── cache/, db/, logging/     # Component configs
│   ├── remote/                   # Remote config (Nacos)
│   └── storage/                  # Storage config
├── constants/                    # Application constants
├── controllers/                  # API route handlers
│   ├── auth/                     # Authentication controllers
│   ├── text_to_image.py          # Image generation endpoints
│   ├── route.py                  # Main API router
│   └── params.py                 # Request/response models
├── diffsynths/                   # DiffSynth integration
│   ├── text_to_image.py          # Text-to-image core logic
│   ├── blog.py                   # Blog/content generation
│   └── mineru.py                 # PDF parsing with MinerU
├── docs/                         # Documentation
│   ├── TEXT_TO_IMAGE.md          # Image generation guide
│   ├── IMAGE_EDITING_API.md      # Image editing guide
│   └── MODEL_MANAGEMENT_FEATURE.md
├── libs/                         # Utility libraries
│   ├── api_key_auth.py           # API key authentication
│   ├── context.py                # Request context/middleware
│   └── deps.py                   # FastAPI dependencies
├── mcp_service/                  # MCP plugin system
├── models/                       # SQLAlchemy ORM models
│   ├── api_key.py                # API key model
│   └── engine.py                 # Database engine setup
├── service/                      # Business logic layer
│   └── api_key_service.py        # API key management
├── test/                         # Test suite
├── utils/                        # Utility functions
│   ├── encoders.py               # JSON encoders
│   ├── http.py                   # HTTP utilities
│   ├── markdown.py               # Markdown processing
│   └── rate_limit.py             # Rate limiting
├── app.py                        # Main FastAPI application
├── app_factory.py                # Application factory pattern
├── aduib_app.py                  # AduibAIApp class definition
├── gradio_app.py                 # Gradio UI application
├── fast_mcp.py                   # FastMCP integration (1069 lines)
├── nacos_mcp.py                  # Nacos MCP wrapper
├── pyproject.toml                # Project metadata & dependencies
├── .ruff.toml                    # Ruff linter configuration
├── .python-version               # Python version (3.12)
├── .env.example                  # Environment template
└── uv.lock                       # Dependency lock file
```

### Key Entry Points

1. **Main FastAPI App**: `app.py` → `app_factory.py` → `aduib_app.py`
   - Creates FastAPI app with MCP integration
   - Initializes middleware (auth, logging, tracing)
   - Registers API routes from `controllers/route.py`

2. **Gradio UI**: `gradio_app.py`
   - Builds multi-tab interface
   - Integrates text-to-image, PDF processing, blog features

3. **Image Generation**: `diffsynths/text_to_image.py`
   - `TextToImageGenerator` class with multi-model support
   - Model loading/unloading for memory management
   - Called by `controllers/text_to_image.py` API endpoints

### Configuration System

**Multi-source configuration** (priority order):
1. Environment variables (`.env` file)
2. Remote settings (Nacos, if enabled)
3. Default values in config classes

**Main config**: `configs/app_config.py`
- Uses Pydantic Settings with custom sources
- `RemoteSettingsSourceFactory` for Nacos integration
- All config classes inherit from `BaseSettings`

### API Router Structure

All API endpoints are registered in `controllers/route.py`:
```python
api_router.include_router(text_to_image_router, prefix="/text-to-image", tags=["Text to Image"])
# Add other routers similarly
```

**API prefix**: `/api` (configured in route registration)

## Common Development Tasks

### Adding a New API Endpoint

1. Create controller in `controllers/your_feature.py`:
   ```python
   from fastapi import APIRouter
   router = APIRouter()
   
   @router.post("/your-endpoint")
   async def your_function():
       pass
   ```

2. Register in `controllers/route.py`:
   ```python
   from controllers.your_feature import router as your_router
   api_router.include_router(your_router, prefix="/your-feature", tags=["Your Feature"])
   ```

3. Add request/response models in `controllers/params.py` (optional)

### Adding a Database Model

1. Create model in `models/your_model.py`:
   ```python
   from models.base import Base  # If base class exists
   from sqlalchemy import Column, Integer, String
   
   class YourModel(Base):
       __tablename__ = "your_table"
       id = Column(Integer, primary_key=True)
   ```

2. Generate migration:
   ```bash
   alembic -c ./alembic/alembic.ini revision --autogenerate -m "add your_table"
   ```

3. Apply migration:
   ```bash
   alembic -c ./alembic/alembic.ini upgrade head
   ```

### Modifying Dependencies

1. Edit `pyproject.toml` dependencies section
2. Delete `uv.lock` (important for clean rebuild)
3. Re-run: `uv sync --dev --index-url https://pypi.org/simple/`

### Debugging Import Errors

If you see `ModuleNotFoundError`:
1. Verify `.venv` is activated: `which python` should show `.venv/bin/python`
2. Reinstall: `uv sync --dev --index-url https://pypi.org/simple/`
3. Check PYTHONPATH doesn't interfere
4. Ensure you're running from repo root directory

## Important Coding Patterns

### Logging

**DO NOT use f-strings in log statements** (ruff rule G004):
```python
# ❌ WRONG
log.info(f"Processing {count} items")

# ✅ CORRECT
log.info("Processing %s items", count)
```

### Memory Management for Image Generation

**Always unload models after generation** to free VRAM:
```python
from diffsynths.text_to_image import get_generator

generator = get_generator()
generator.load_model("sd")
image_path = generator.generate(prompt="...")
generator.unload_model()  # Important!
```

### Low VRAM Mode

When `low_vram=True`, apply `vram_config` to **ALL** ModelConfig instances:
```python
# From repository memory: Apply vram_config to all configs including tokenizer_config
if low_vram:
    model_configs = [
        ModelConfig(..., **vram_config),
        ModelConfig(..., **vram_config),  # Don't forget any configs!
    ]
```

### API Key Authentication

If `AUTH_ENABLED=True` in `.env`, all API requests require `X-API-Key` header:
```bash
curl -H "X-API-Key: your-key-here" http://localhost:5002/api/endpoint
```

## Continuous Integration / Pre-commit Checks

**No CI workflows configured** - This repository does not have `.github/workflows/` directory.

**Manual pre-commit validation**:
```bash
# 1. Lint code
ruff check . --fix

# 2. Run tests (if applicable)
pytest

# 3. Check types (if using mypy - not currently configured)
# mypy .
```

**Before committing**:
- Run `ruff check . --fix` to auto-fix import sorting and simple issues
- Manually fix any remaining ruff errors (especially line length)
- Ensure changes don't break existing tests

## Troubleshooting Guide

### Problem: `uv sync` fails with DNS error on Aliyun mirror
**Solution**: See "Known Build Issue" section above - remove `uv.lock` and use PyPI index override

### Problem: `ImportError: cannot import name 'ModelManager' from 'diffsynth'`
**Solution**: Install optional DiffSynth dependency:
```bash
uv sync --extra diffsynth --index-url https://pypi.org/simple/
```

### Problem: `RuntimeError: CUDA out of memory`
**Solution**:
- Reduce image dimensions (e.g., 512x512 instead of 1024x1024)
- Reduce inference steps (e.g., 20 instead of 50)
- Use `low_vram=True` when loading models
- Unload models after generation
- Use CPU mode (set `device="cpu"` - will be slow)

### Problem: Alembic migration fails
**Solution**:
- Verify database connection in `.env`
- Check PostgreSQL is running: `psql -h <host> -U <user> -d <database>`
- Ensure database exists: `createdb aduib_ai` (if needed)
- Run migrations one at a time if multiple pending

### Problem: Tests fail with import errors
**Solution**:
- Reinstall all dependencies: `uv sync --dev --index-url https://pypi.org/simple/`
- Activate venv: `source .venv/bin/activate`
- Run from repo root: `cd /home/runner/work/aduib-diffsynth/aduib-diffsynth && pytest`

### Problem: Application fails to start with "No module named 'configs'"
**Solution**:
- You're likely not in the repo root directory
- Run from: `/home/runner/work/aduib-diffsynth/aduib-diffsynth/`
- Ensure `.venv` is activated

### Problem: Port already in use
**Solution**:
```bash
# Find process using port 5002
lsof -i :5002
# Kill it (or change APP_PORT in .env)
kill -9 <PID>
```

## Performance Notes

- **First run**: App initialization takes ~1-2 seconds (logging, middleware setup)
- **Model loading**: First-time model download can be several GB, subsequent loads faster
- **Image generation**: 5-30 seconds depending on model, size, steps, hardware
- **Database queries**: Generally <100ms with proper indexing

## Additional Resources

- **README.md**: Basic project introduction and quick start
- **docs/TEXT_TO_IMAGE.md**: Comprehensive text-to-image documentation with examples
- **docs/IMAGE_EDITING_API.md**: Image editing feature documentation
- **.env.example**: All available environment variables with descriptions

---

## Agent Instructions

**Trust these instructions** - They were created by thoroughly exploring and testing the repository. Only search for additional information if:
- These instructions are incomplete for your specific task
- You encounter errors not covered in the troubleshooting section
- You need to understand implementation details not covered here

**Always start by**:
1. Running `uv sync --dev --index-url https://pypi.org/simple/` (if dependencies changed)
2. Activating venv: `source .venv/bin/activate`
3. Working from repo root: `/home/runner/work/aduib-diffsynth/aduib-diffsynth/`

**When making changes**:
- Run `ruff check . --fix` before committing
- Test affected functionality manually (no comprehensive test coverage)
- Document any new workarounds you discover
