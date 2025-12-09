from pydantic import Field
from pydantic_settings import BaseSettings


class AduibAiConfig(BaseSettings):

    ADUIB_SERVICE_URL: str = Field(default="https://aduib.ai",description="Aduib service url")
    ADUIB_SERVICE_TOKEN: str = Field(default="",description="Aduib service token")
    ADUIB_SERVICE_TIMEOUT: int = Field(default=300,description="Aduib service timeout")