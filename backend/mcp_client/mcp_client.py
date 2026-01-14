import httpx
from backend.config.config import get_settings
import logging

settings = get_settings()
logger = logging.getLogger("sentinel.mcp")

class MCPClient:
    def __init__(self):
        self.base_url = settings.MCP_SERVER_URL
        self.client = httpx.AsyncClient(timeout=5.0)

    async def send_context(self, context_data: dict):
        """Send detection context to MCP server for higher-level reasoning."""
        try:
            response = await self.client.post(f"{self.base_url}/context", json=context_data)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as exc:
            logger.warning(f"An error occurred while requesting {exc.request.url!r}.")
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            return None

    async def close(self):
        await self.client.aclose()

mcp_client = MCPClient()
