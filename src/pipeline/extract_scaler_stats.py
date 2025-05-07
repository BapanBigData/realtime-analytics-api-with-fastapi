import pandas as pd
from src.config.schema import PreprocessingConfig

def extract_scaler_stats(pipeline):
    preprocessor_fitted = pipeline.named_steps['preprocessor']
    scaler_fitted = preprocessor_fitted.named_transformers_['scale']

    mu = scaler_fitted.mean_
    sigma = scaler_fitted.scale_

    config = PreprocessingConfig()
    scaled_features = config.features_to_scale

    scaler_stats = pd.DataFrame({'feature': scaled_features, 'mu (mean)': mu, 'sigma (std)': sigma})
    return scaler_stats
