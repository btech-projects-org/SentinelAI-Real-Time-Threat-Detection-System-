
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from backend.config.config import get_settings

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/token")

async def get_current_user():
    """
    Validate the JWT token and return the current user (subject).
    DEV MODE: AUTHENTICATION DISABLED - ALWAYS RETURNS 'admin'
    """
    # For local development where we want to skip auth
    return "admin"
    # For local development where we want to skip auth
    return "admin"
    
    # ORIGINAL AUTH LOGIC (COMMENTED OUT)
    # credentials_exception = HTTPException(
    #     status_code=status.HTTP_401_UNAUTHORIZED,
    #     detail="Could not validate credentials",
    #     headers={"WWW-Authenticate": "Bearer"},
    # )
    # try:
    #     payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
    #     username: str = payload.get("sub")
    #     if username is None:
    #         raise credentials_exception
    # except JWTError:
    #     raise credentials_exception
        
    # if username != "admin":
    #     raise credentials_exception
        
    # return username
