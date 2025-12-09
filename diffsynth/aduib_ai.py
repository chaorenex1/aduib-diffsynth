import logging
from typing import Optional

from configs import config
from utils.http import BaseHTTPClient

logger = logging.getLogger(__name__)


class AduibAIClient(BaseHTTPClient):
    """带认证的 AduibAI API 客户端"""

    def __init__(self):
        """初始化 AduibAI 客户端。"""
        super().__init__(
            base_url=config.ADUIB_SERVICE_URL,
            timeout=config.ADUIB_SERVICE_TIMEOUT,
        )
        self._authenticated = False

    def authenticate(self) -> None:
        """
        与 AduibAI 服务器进行认证。

        异常:
            ConfigurationError：未配置认证方式
            AuthenticationError：认证失败
        """
        # Try token authentication first
        token = config.ADUIB_SERVICE_TOKEN
        if token:  # Type guard to ensure token is not None
            self.set_auth_token(token)
            self._authenticated = True
            logger.info("使用令牌认证成功")
            return

    def ensure_authenticated(self) -> None:
        """确保客户端已通过认证。"""
        if not self._authenticated:
            self.authenticate()


# Global aduib_ai client instance
aduib_ai_client: Optional[AduibAIClient] = None


def get_aduib_ai_client() -> AduibAIClient:
    """获取或创建 aduib_ai 客户端实例。"""
    global aduib_ai_client
    if aduib_ai_client is None:
        aduib_ai_client = AduibAIClient()
        # settings.halo_base_url = config.HALO_BASE_URL
        # settings.halo_token = config.HALO_API_KEY
        # settings.mcp_timeout = config.HALO_TIMEOUT
        aduib_ai_client.connect()
        aduib_ai_client.authenticate()
        logger.info("aduib_ai 客户端已初始化")
    return aduib_ai_client


"""
@router.post("/knowledge/rag/paragraph")
@catch_exceptions
async def create_paragraph_rag(file: UploadFile = File(...)):
    await KnowledgeBaseService.paragraph_rag_from_blog_content(await file.read())
    return BaseResponse.ok()
"""
def create_paragraph_rag(file_content: bytes, file_name: str) -> dict:
    """
    创建段落 RAG。

    参数:
        file_content: 文件内容字节
        file_name: 文件名

    返回:
        API 响应结果

    异常:
        RuntimeError: 请求失败时抛出
    """
    client = get_aduib_ai_client()
    client.ensure_authenticated()

    # 准备文件上传 - 使用 BytesIO 包装文件内容
    from io import BytesIO

    # 创建文件类对象
    file_obj = BytesIO(file_content)

    # 准备文件上传 - httpx 需要的格式：(filename, file_object, content_type)
    files = {
        "file": (file_name, file_obj, "text/markdown")
    }

    try:
        # 发起请求
        response = client.post("/v1/knowledge/rag/paragraph", files=files,data=None,json=None)
        logger.info(f"段落 RAG 创建成功: {file_name}")
        return response
    finally:
        # 确保关闭 BytesIO 对象
        file_obj.close()
