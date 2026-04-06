from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

def register_error_handlers(app: FastAPI):
    @app.exception_handler(Exception)
    async def global_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500,
            content={"error": "Internal Server Error", "detail": str(exc)})
