from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="분석할 한국어 텍스트")


class SentimentResult(BaseModel):
    label: str
    score: float


class SentimentResponse(BaseModel):
    success: bool
    result: SentimentResult
    message: str
