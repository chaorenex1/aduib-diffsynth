"""带重试与错误处理的基础 HTTP 客户端"""
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    """通用功能的基础 HTTP 客户端"""

    def __init__(self, base_url: str, timeout: int = 30):
        """
        初始化 HTTP 客户端。

        参数:
            base_url: API 基础地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None
        self._headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc_info):
        self.close()

    def connect(self) -> None:
        """创建 HTTP 客户端连接。"""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
                # headers=self._headers,
                follow_redirects=True,
            )
            logger.debug(f"HTTP 客户端已连接：{self.base_url}")

    def close(self) -> None:
        """关闭 HTTP 客户端连接。"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug("HTTP 客户端已关闭")

    def set_auth_token(self, token: str) -> None:
        """
        设置认证令牌。

        参数:
            token: Bearer 令牌
        """
        self._headers["Authorization"] = f"Bearer {token}"
        if self._client:
            self._client.headers.update({"Authorization": f"Bearer {token}"})
        logger.debug("认证令牌已设置")

    def remove_auth_token(self) -> None:
        """移除认证令牌。"""
        self._headers.pop("Authorization", None)
        if self._client:
            self._client.headers.pop("Authorization", None)
        logger.debug("认证令牌已移除")

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        发起带重试逻辑的 HTTP 请求。

        参数:
            method: HTTP 方法
            path: 请求路径
            params: 查询参数
            json: JSON 请求体
            data: 表单数据
            files: 上传文件
            headers: 额外请求头

        返回:
            响应 JSON

        异常:
            AuthenticationError：认证失败
            ResourceNotFoundError：资源未找到
            NetworkError：网络/HTTP 错误
        """
        if not self._client:
            self.connect()

        url = path if path.startswith("http") else f"{self.base_url}{path}"
        request_headers = {**self._headers, **(headers or {})}

        # Remove Content-Type for multipart/form-data (httpx will set it)
        if files:
            request_headers.pop("Content-Type", None)
            request_headers.pop("Accept", None)

        retry_count = 0
        last_error = None

        while retry_count <= 3:
            try:
                logger.debug(f"API 请求：{method} {url}")

                response = self._client.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                    headers=request_headers,
                )

                # Handle error status codes
                if response.status_code == 401:
                    logger.error("认证失败（401）")
                    raise RuntimeError("认证失败。请检查您的凭据。")

                if response.status_code == 403:
                    logger.error("权限不足（403）")
                    raise RuntimeError("权限不足。您没有访问该资源的权限。")

                if response.status_code == 404:
                    logger.error(f"资源未找到（404）：{url}")
                    raise RuntimeError(f"资源未找到：{url}")

                if response.status_code >= 500:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = (
                            error_json.get("detail") or error_json.get("message") or error_detail
                        )
                    except Exception:
                        pass

                    logger.error(f"HTTP {response.status_code} 错误：{error_detail}")
                    raise RuntimeError(f"HTTP {response.status_code} 错误：{error_detail}")

                if response.status_code >= 400:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = (
                            error_json.get("detail") or error_json.get("message") or error_detail
                        )
                    except Exception:
                        pass

                    logger.error(f"HTTP {response.status_code} 错误：{error_detail}")
                    raise RuntimeError(f"HTTP {response.status_code} 错误：{error_detail}")

                # Parse response
                if response.status_code == 204:
                    return {}

                try:
                    result = response.json()
                    logger.debug(f"API 响应：{response.status_code}")
                    return result
                except Exception as e:
                    logger.warning(f"解析 JSON 响应失败：{e}")
                    return {"text": response.text}

            except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
                last_error = e
                retry_count += 1
                if retry_count <= 3:
                    logger.warning(
                        f"请求失败，正在重试（{retry_count}/3）：{e}"
                    )
                    self._wait_retry()
                else:
                    logger.error(f"请求在重试 3 次后仍失败：{e}")
                    raise RuntimeError(f"请求失败：{e}")

            except RuntimeError as e:
                logger.error(f"请求失败：{e}")
                raise

            except Exception as e:
                logger.error(f"请求过程中出现未预期错误：{e}", exc_info=True)
                raise RuntimeError(f"请求过程中出现未预期错误：{e}")

        raise RuntimeError(f"请求失败：{last_error}")

    def _wait_retry(self) -> None:
        """重试前等待。"""
        import time

        time.sleep(2)

    def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """GET 请求。"""
        return self._request("GET", path, params=params, headers=headers)

    def post(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """POST 请求。"""
        return self._request(
            "POST", path, params=params, json=json, data=data, files=files, headers=headers
        )

    def put(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """PUT 请求。"""
        return self._request("PUT", path, params=params, json=json, headers=headers)

    def patch(
        self,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """PATCH 请求。"""
        return self._request("PATCH", path, params=params, json=json, headers=headers)

    def delete(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """DELETE 请求。"""
        return self._request("DELETE", path, params=params, headers=headers)

    @property
    def headers(self):
        return self._headers
