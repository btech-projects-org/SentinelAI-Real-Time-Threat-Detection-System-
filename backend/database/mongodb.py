from motor.motor_asyncio import AsyncIOMotorClient
from backend.config.config import get_settings
import logging
import asyncio

settings = get_settings()
logger = logging.getLogger("sentinel.database")

class Database:
    client: AsyncIOMotorClient = None
    db = None

    async def connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            self.db = self.client[settings.DB_NAME]
            
            # Verify connection with timeout
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=5.0
            )
            
            logger.info(f"✅ Connected to MongoDB: {settings.DB_NAME}")
            
        except asyncio.TimeoutError:
            error_msg = "MongoDB connection timeout - database unavailable"
            logger.critical(error_msg)
            self.client = None
            self.db = None
            # In production, you might want to raise here to prevent app start
            # raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"MongoDB Connection Failed: {e}"
            logger.critical(error_msg)
            self.client = None
            self.db = None
            # In production, you might want to raise here
            # raise RuntimeError(error_msg)

    async def close(self):
        """Close connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB Connection Closed")

    async def save_alert(self, alert_data: dict):
        """Save a detection alert to the database."""
        if self.db is not None:
            try:
                result = await self.db.alerts.insert_one(alert_data)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Failed to save alert: {e}")
                return None
        return None

    async def get_recent_alerts(self, limit: int = 50):
        """Retrieve recent alerts."""
        if self.db is not None:
            try:
                cursor = self.db.alerts.find().sort("timestamp", -1).limit(limit)
                return await cursor.to_list(length=limit)
            except Exception as e:
                logger.error(f"Failed to fetch alerts: {e}")
                return []
        return []

    async def save_criminal(self, criminal_data: dict):
        """Save a criminal profile to the database."""
        if self.db is not None:
            try:
                # Check if criminal already exists by ID
                existing = await self.db.criminals.find_one({"criminal_id": criminal_data.get("criminal_id")})
                if existing:
                    result = await self.db.criminals.update_one(
                        {"criminal_id": criminal_data.get("criminal_id")},
                        {"$set": criminal_data}
                    )
                    logger.info(f"Updated criminal: {criminal_data.get('criminal_id')}")
                    return str(existing.get("_id"))
                else:
                    result = await self.db.criminals.insert_one(criminal_data)
                    logger.info(f"Saved new criminal: {criminal_data.get('criminal_id')}")
                    return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Failed to save criminal: {e}")
                return None
        return None

    async def get_all_criminals(self):
        """Retrieve all registered criminals."""
        if self.db is not None:
            try:
                cursor = self.db.criminals.find({})
                criminals = await cursor.to_list(length=1000)
                logger.info(f"Retrieved {len(criminals)} criminals from database")
                return criminals
            except Exception as e:
                logger.error(f"Failed to fetch criminals: {e}")
                return []
        return []

    async def get_criminal_by_id(self, criminal_id: str):
        """Get a specific criminal by ID."""
        if self.db is not None:
            try:
                criminal = await self.db.criminals.find_one({"criminal_id": criminal_id})
                return criminal
            except Exception as e:
                logger.error(f"Failed to fetch criminal {criminal_id}: {e}")
                return None
        return None

    async def save_telegram_message(self, message_data: dict):
        """Save a telegram message/notification to the database."""
        if self.db is not None:
            try:
                result = await self.db.telegram_messages.insert_one(message_data)
                logger.info(f"Saved telegram message: {message_data.get('message_id')}")
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Failed to save telegram message: {e}")
                return None
        return None

    async def get_telegram_messages(self, limit: int = 50, chat_id: str = None):
        """Retrieve telegram messages, optionally filtered by chat_id."""
        if self.db is not None:
            try:
                query = {"chat_id": chat_id} if chat_id else {}
                cursor = self.db.telegram_messages.find(query).sort("timestamp", -1).limit(limit)
                messages = await cursor.to_list(length=limit)
                return messages
            except Exception as e:
                logger.error(f"Failed to fetch telegram messages: {e}")
                return []
        return []

    async def update_telegram_message_status(self, message_id: str, status: str):
        """Update the status of a telegram message."""
        if self.db is not None:
            try:
                result = await self.db.telegram_messages.update_one(
                    {"message_id": message_id},
                    {"$set": {"status": status}}
                )
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"Failed to update telegram message status: {e}")
                return False
        return False

    async def get_telegram_stats(self):
        """Get statistics about telegram messages."""
        if self.db is not None:
            try:
                total_messages = await self.db.telegram_messages.count_documents({})
                sent_messages = await self.db.telegram_messages.count_documents({"status": "sent"})
                failed_messages = await self.db.telegram_messages.count_documents({"status": "failed"})
                
                return {
                    "total_messages": total_messages,
                    "sent": sent_messages,
                    "failed": failed_messages,
                    "pending": total_messages - sent_messages - failed_messages
                }
            except Exception as e:
                logger.error(f"Failed to get telegram stats: {e}")
                return None
        return None

db = Database()
