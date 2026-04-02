from pydantic import BaseModel, Field

class HousingRequest(BaseModel):
    MedInc:     float = Field(..., gt=0, description="중위 소득 (만달러)")
    HouseAge:   float = Field(..., ge=0, le=100, description="주택 연식 (년)")
    AveRooms:   float = Field(..., gt=0, description="평균 방 수")
    AveBedrms:  float = Field(..., gt=0, description="평균 침실 수")
    Population: float = Field(..., gt=0, description="인구")
    AveOccup:   float = Field(..., gt=0, description="평균 거주자 수")
    Latitude:   float = Field(..., ge=32, le=42, description="위도")
    Longitude:  float = Field(..., ge=-125, le=-114, description="경도")

class HousingResponse(BaseModel):
    predicted_price: float
    predicted_price_unit: str
    confidence_note: str
