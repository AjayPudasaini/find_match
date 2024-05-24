import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from typing import List
from database.db import SessionLocal

def fetch_data_from_db(db: Session) -> pd.DataFrame:
    query = """
            SELECT
                CONCAT(first_name, ' ', middle_name, ' ', last_name) as "Name",
                citizenship_no,
                TO_CHAR(date_of_birth,'YYYY-MM-DD') as Date_of_birth,
                CONCAT(father_first_name, ' ', father_middle_name, ' ', father_last_name) as "Father_Name"
            FROM kycn_personal_info
        """
    result = db.execute(query)
    data = result.fetchall()
    columns = result.keys()

    # Convert the result to a DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Replace None and empty strings with NaN
    df.replace({None: np.nan, '': np.nan}, inplace=True)
    
    df.rename(columns={
        'citizenship_no': 'Citizenship_no',
        'date_of_birth': 'Date_of_birth'
    }, inplace=True)


    print(df.head(5))

    return df


def datas():
    with SessionLocal() as db:
        df = fetch_data_from_db(db)
        return df
