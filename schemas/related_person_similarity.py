from pydantic import BaseModel
from datetime import datetime

class RelatedPersonSimilarityCreate(BaseModel):
    primary_data_id: int
    similar_data_id: int
    weighted_similarity: float

class RelatedPersonSimilarityResponse(BaseModel):
    id: int
    primary_data_id: int
    similar_data_id: int
    weighted_similarity: float

    class Config:
        orm_mode = True