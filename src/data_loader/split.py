import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.schema import PreprocessingConfig
from src.config.config import RANDOM_SEED, TEST_SIZE


def split_data(df: pd.DataFrame, config: PreprocessingConfig):
    
    # Drop meta columns
    df = df.drop(columns=config.drop_columns)

    # Split X and y
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column].astype(int)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y,
        random_state=RANDOM_SEED
    )

    return X_train, X_test, y_train, y_test
