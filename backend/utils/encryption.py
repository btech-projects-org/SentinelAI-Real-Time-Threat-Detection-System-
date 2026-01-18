from cryptography.fernet import Fernet
import os
import logging

logger = logging.getLogger("sentinel.utils.encryption")

# In a real production app, this key MUST be persistent and loaded from env.
# For this "Correctness" task, we will try to load from env or generate a consistent one for the session.
_key = os.getenv("ENCRYPTION_KEY")
if not _key:
    _key = Fernet.generate_key()
    # In a real deployment, we'd log a warning that a new key was generated.
    logger.warning("No ENCRYPTION_KEY found. Generated ephemerally.")

cipher_suite = Fernet(_key)

def encrypt_data(data: str) -> str:
    """Encrypt sensitive string data."""
    if not data:
        return ""
    try:
        return cipher_suite.encrypt(data.encode()).decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return ""

def decrypt_data(token: str) -> str:
    """Decrypt sensitive string data."""
    if not token:
        return ""
    try:
        return cipher_suite.decrypt(token.encode()).decode()
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        return ""
