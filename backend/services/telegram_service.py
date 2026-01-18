"""
Telegram Bot Service for sending alerts and notifications
"""
import aiohttp
import logging
from typing import Optional
from backend.config.config import get_settings
from datetime import datetime

logger = logging.getLogger("sentinel.services.telegram")

class TelegramService:
    def __init__(self):
        settings = get_settings()
        self.bot_token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID
        self.enabled = settings.TELEGRAM_ENABLED
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if self.enabled:
            logger.info("✅ Telegram Bot Service Initialized")
        else:
            logger.warning("⚠️  Telegram Bot Service Disabled - No credentials provided")

    def update_credentials(self, bot_token: str, chat_id: str):
        """Update Telegram credentials dynamically at runtime."""
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        
        if self.enabled:
            logger.info("🔄 Telegram credentials updated successfully")
        else:
            logger.warning("⚠️  Telegram disabled (empty credentials)")
    
    async def send_message(self, text: str) -> bool:
        """Send a text message to Telegram chat."""
        if not self.enabled:
            logger.warning("Telegram not enabled - skipping message")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML"
                }
                
                async with session.post(f"{self.api_url}/sendMessage", json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ Telegram message sent successfully")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Failed to send Telegram message: {error}")
                        return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False
    
    async def send_alert(self, alert_data: dict) -> bool:
        """Send a formatted threat alert to Telegram."""
        if not self.enabled:
            return False
        
        try:
            # Build alert message
            message = self._format_alert(alert_data)
            return await self.send_message(message)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    async def send_detection_alert(
        self,
        detection_type: str,
        severity: str,
        confidence: float,
        location: Optional[str] = None,
        criminal_id: Optional[str] = None,
        criminal_name: Optional[str] = None
    ) -> bool:
        """Send a detection alert with structured format."""
        if not self.enabled:
            return False
        
        severity_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢"
        }
        
        emoji = severity_emoji.get(severity, "⚠️")
        
        message = f"""
{emoji} <b>THREAT DETECTION ALERT</b>

<b>Type:</b> {detection_type}
<b>Severity:</b> {severity}
<b>Confidence:</b> {confidence:.1%}
<b>Time:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        
        if location:
            message += f"<b>Location:</b> {location}\n"
        
        if criminal_id:
            message += f"<b>Criminal ID:</b> {criminal_id}\n"
        
        if criminal_name:
            message += f"<b>Criminal:</b> {criminal_name}\n"
        
        return await self.send_message(message)
    
    def _format_alert(self, alert_data: dict) -> str:
        """Format alert data into HTML message."""
        message = "<b>🚨 THREAT ALERT</b>\n\n"
        
        if "label" in alert_data:
            message += f"<b>Type:</b> {alert_data['label']}\n"
        
        if "confidence" in alert_data:
            message += f"<b>Confidence:</b> {alert_data['confidence']:.1%}\n"
        
        if "timestamp" in alert_data:
            message += f"<b>Time:</b> {alert_data['timestamp']}\n"
        
        if "description" in alert_data:
            message += f"<b>Description:</b> {alert_data['description']}\n"
        
        return message
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        if not self.enabled:
            logger.warning("Telegram not configured")
            return False
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/getMe") as response:
                    if response.status == 200:
                        data = await response.json()
                        bot_name = data.get("result", {}).get("username", "Unknown")
                        logger.info(f"✅ Telegram bot connected: @{bot_name}")
                        return True
                    else:
                        logger.error("Failed to connect to Telegram API")
                        return False
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False


# Global instance
telegram_service = TelegramService()
