from fastapi import Header, HTTPException

VALID_KEYS = {"test-api-key-1234"}

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key
