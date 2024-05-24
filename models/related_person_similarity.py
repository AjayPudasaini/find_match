from database.db import Base
from sqlalchemy import Column, Integer, DateTime, Float
from sqlalchemy.sql import func



class RelatedPersonSimilarity(Base):
    __tablename__ = "related_person_similarity"
    id = Column(Integer, primary_key=True, index=True)
    primary_data_id = Column(Integer)
    similar_data_id = Column(Integer)
    weighted_similarity = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
