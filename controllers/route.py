from fastapi import APIRouter

from .auth import api_key
from .text_to_image import router as text_to_image_router

api_router = APIRouter()

#auth
api_router.include_router(api_key.router)

#text-to-image
api_router.include_router(text_to_image_router)
