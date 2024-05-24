from typing import List
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session
from fastapi import APIRouter

from database.db import get_db
from schemas.related_person_similarity import RelatedPersonSimilarityCreate, RelatedPersonSimilarityResponse
from models.related_person_similarity import RelatedPersonSimilarity as RelatedPersonSimilarityModel
from ml_model.ml_model import get_final_result_df
from pydantic import parse_obj_as


router = APIRouter(
    prefix="/find-match",
    tags=["Find Match"]
)


def save_data_to_db(data: List[RelatedPersonSimilarityCreate], db: Session):
    for item in data:
        new_entry = RelatedPersonSimilarityModel(
            primary_data_id=item.primary_data_id,
            similar_data_id=item.similar_data_id
        )
        db.add(new_entry)
    db.commit()

@router.get("/", response_model=List[RelatedPersonSimilarityResponse])
def find_match(db: Session = Depends(get_db)):
    try:
        results = get_final_result_df()
        # Validate and parse the data using Pydantic
        parsed_results = parse_obj_as(List[RelatedPersonSimilarityCreate], results)
        # Save the parsed data to the database
        save_data_to_db(parsed_results, db)
        
        # Retrieve the saved data to return in the response
        saved_data = db.query(RelatedPersonSimilarityModel).filter(
            RelatedPersonSimilarityModel.primary_data_id.in_([result.primary_data_id for result in parsed_results]),
            RelatedPersonSimilarityModel.similar_data_id.in_([result.similar_data_id for result in parsed_results])
        ).all()

        return saved_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
