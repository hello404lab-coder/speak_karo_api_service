"""In-process WebSocket registry for social matchmaking (single-worker assumption)."""
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}

    def register(self, user_id: str, websocket: WebSocket) -> None:
        """Register an already-accepted WebSocket for this user."""
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str) -> None:
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_to_user(self, user_id: str, message: dict) -> bool:
        ws = self.active_connections.get(user_id)
        if not ws:
            logger.warning("social_ws_send_skip user_id=%s (no active connection)", user_id)
            return False
        try:
            await ws.send_json(message)
            return True
        except Exception as e:
            logger.warning("social_ws_send_failed user_id=%s: %s", user_id, e)
            return False


manager = ConnectionManager()
