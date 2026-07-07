from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from FG.core.config import JWT_SECRET_KEY, JWT_ALGORITHM,EXPECTED_USER_ID,EXPECTED_ROLE

security = HTTPBearer()
#JWT_SECRET_KEY='Fluent@dec27'
#JWT_ALGORITHM='HS256'
#user_id='69d8bb2e-a0a06073-294c684c-2a7a5dcf'
#role ='ROLE_BIHAR'

def create_jwt_token(expiration_minutes: int = 60):

    payload = {
        "userId": EXPECTED_USER_ID,
        "role": EXPECTED_ROLE,
    }
    header = {
        "alg": JWT_ALGORITHM
        # You can omit 'typ' here if you don't want it
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, headers=header)
    print("Created Token:", token)
    print("Secret Used (create):", JWT_SECRET_KEY)    
    return token


def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

# 2. Verify token
def verify_test_token(token: str):
    try:
        
        decoded = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        print("✅ Token is valid!")
        print("Decoded Payload:", decoded)
        return decoded
    except JWTError as e:
        print("❌ JWT decode error:", str(e))
        return None

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        print("Received Token:", token)
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        print("JWT decode error:", str(e))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired token",
        )
