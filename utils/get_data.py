import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import List
from database.db import SessionLocal

def fetch_data_from_db(db: Session) -> pd.DataFrame:
    query = """
            SELECT
                id,
                CONCAT(first_name, ' ', middle_name, ' ', last_name) as "Name",
                primary_identification_document_no,
                dob_bs,
                father_name
            FROM kycn_related_person_info
        """
    result = db.execute(query)
    data = result.fetchall()
    columns = result.keys()

    # Convert the result to a DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Replace None and empty strings with NaN
    df.replace({None: np.nan, '': np.nan}, inplace=True)
    
    # Rename columns
    df.rename(columns={
        'primary_identification_document_no': 'Citizenship_no',
        'dob_bs': 'Date_of_birth',
        'father_name': 'Father_Name'
    }, inplace=True)

    print(df.head(5))
    return df

def datas():
    with SessionLocal() as db:
        df = fetch_data_from_db(db)
        return df