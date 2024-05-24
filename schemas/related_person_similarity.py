from pydantic import BaseModel
from datetime import datetime

class RelatedPersonSimilarityCreate(BaseModel):
    primary_data_id: int
    similar_data_id: int

class RelatedPersonSimilarityResponse(BaseModel):
    id: int
    primary_data_id: int
    similar_data_id: int
    created_at: datetime
    updated_at: datetime
