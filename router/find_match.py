# routes/find_match.py
from typing import List
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi import APIRouter
from datetime import datetime
from database.db import get_db
from schemas.related_person_similarity import RelatedPersonSimilarityCreate, RelatedPersonSimilarityResponse
from models.related_person_similarity import RelatedPersonSimilarity as RelatedPersonSimilarityModel
from ml_model.ml_model import final_output

router = APIRouter(
    prefix="/find-match",
    tags=["Find Match"]
)

def save_data_to_db(data: List[RelatedPersonSimilarityCreate], db: Session):
    print("l18", data)
    for item in data:
        new_entry = RelatedPersonSimilarityModel(
            primary_data_id=item.get("primary_customer_id"),
            similar_data_id=item.get("similar_screening_id"),
            weighted_similarity=item.get("weighted_similarity"),
        )
        db.add(new_entry)
    db.commit()

@router.get("/")
def find_match(db: Session = Depends(get_db)):
    try:
        data = final_output
        save_data_to_db(data, db)
        return data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
