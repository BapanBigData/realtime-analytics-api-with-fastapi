import pandas as pd
import numpy as np
from pydantic import ValidationError
from src.config.schema import RaceInput
from src.config.config import RANDOM_SEED
from src.utils.logger import logging


def validate_dataframe(df):
    """
    Validates each row of the DataFrame against the RaceInput schema.
    
    Raises:
        ValidationError if any row does not conform.
    """
    errors = []
    for idx, row in df.iterrows():
        data = row.to_dict()
        try:
            RaceInput(**data)
        except ValidationError as e:
            errors.append((idx, e))

    if errors:
        for idx, err in errors[:5]:  # Display max first 5 errors
            # print(f"Row {idx} validation error:\n{err}")
            logging.info(f"Row {idx} validation error:\n{err}")
        raise ValueError(f"{len(errors)} rows failed schema validation")


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path, engine='fastparquet')
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    logging.info("Data validation starts...")
    
    # Validate against schema
    validate_dataframe(df)
    
    logging.info("Data validation successful!")
    
    logging.info("Imputing the missing values...")
    
    qdelta_95 = np.nanpercentile(df['qualifying_delta'], 95)
    df['qualifying_delta'] = df['qualifying_delta'].fillna(qdelta_95)
    
    logging.info("Missing value imputation done!")
    
    return df
