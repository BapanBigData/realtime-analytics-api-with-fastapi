from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.config.schema import PreprocessingConfig
from src.config.config import RANDOM_SEED


def build_pipeline():
    config = PreprocessingConfig()

    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), config.features_to_scale),
            ('keep', 'passthrough', config.features_to_keep)
        ]
    )

    xg_model = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric='logloss'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xg_model)
    ])

    return pipeline
