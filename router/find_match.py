# routes/find_match.py
from typing import List
from fastapi import HTTPException, Depends, APIRouter
from sqlalchemy.orm import Session
from database.db import get_db
from schemas.related_person_similarity import RelatedPersonSimilarityCreate, RelatedPersonSimilarityResponse
from models.related_person_similarity import RelatedPersonSimilarity as RelatedPersonSimilarityModel
from ml_model.ml_model import final_output

router = APIRouter(
    prefix="/find-match",
    tags=["Find Match"]
)

def save_data_to_db(data: List[RelatedPersonSimilarityCreate], db: Session):
    # print("l18", data)
    for item in data:
        new_entry = RelatedPersonSimilarityModel(
            primary_data_id=int(item.get("primary_customer_id")),
            similar_data_id=int(item.get("similar_screening_id")),
            weighted_similarity=float(item.get("weighted_similarity")),
        )
        db.add(new_entry)
    db.commit()

def convert_to_standard_types(data: List[dict]) -> List[dict]:
    converted_data = []
    for item in data:
        converted_item = {
            "primary_customer_id": int(item["primary_customer_id"]),
            "similar_screening_id": int(item["similar_screening_id"]),
            "weighted_similarity": float(item["weighted_similarity"])
        }
        converted_data.append(converted_item)
    return converted_data

@router.get("/")
def find_match(db: Session = Depends(get_db)):
    try:
        data = final_output
        converted_data = convert_to_standard_types(data)
        save_data_to_db(converted_data, db)
        return converted_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
