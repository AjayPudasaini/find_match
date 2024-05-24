from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DATABASE_URL = "postgresql://data_cleaning:data_cleaning%40123@172.16.99.100:5432/data_cleaning"
# DATABASE_URL = "postgresql://ajay:1530ajay@localhost/rawa"


DATABASE_URL = "postgresql://postgres:datum%40123@202.166.198.129:5080/taml_datum2"
print("l10", DATABASE_URL)
engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()