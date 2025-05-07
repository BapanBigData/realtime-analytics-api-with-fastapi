import time
from src.data_loader.loader import load_training_data
from src.data_loader.split import split_data
from src.data_loader.smote import apply_smote
from src.pipeline.build_pipeline import build_pipeline
from src.pipeline.extract_scaler_stats import extract_scaler_stats
from src.config.schema import PreprocessingConfig
from src.utils.helpers import save_model
from src.utils.logger import logging
from src.model.performance_metrics import evaluate_model
from src.config.config import MODEL_DIR, SCALER_STATS_DIR, TRAIN_DATA_PATH


MODEL_SAVE_PATH = MODEL_DIR / 'pipeline.joblib'
STATS_SAVE_PATH = SCALER_STATS_DIR / 'scaler_stats.csv'


def train(training_data_path: str):
    logging.info("Loading training data...")
    df = load_training_data(training_data_path)
    
    config = PreprocessingConfig()
    
    logging.info("Preprocessing and split the data...")
    X_train, X_test, y_train, y_test = split_data(df, config)

    logging.info("Applying SMOTE on training data...")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    logging.info("Initializing the ML pipeline and train the model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train_smote, y_train_smote)

    logging.info("Extracting the scaler stats...")
    scaler_stats = extract_scaler_stats(pipeline)
    scaler_stats.to_csv(STATS_SAVE_PATH, index=False)

    logging.info("Save the full pipeline (preprocessor + XGBoost)")
    save_model(pipeline, MODEL_SAVE_PATH)

    logging.info(f"✅ Model trained and saved to {MODEL_SAVE_PATH}!")
    logging.info(f"✅ Scaler stats saved to {STATS_SAVE_PATH}!")
    
    logging.info("Model evalution starts...")
    
    f1_score_minority_class = round(evaluate_model(pipeline, X_test, y_test), 4)
    logging.info(f"f1-score of the minority class is: {f1_score_minority_class}")
    
    logging.info("Training ends!")

    return 


if __name__ == "__main__":
    try:
        start_time = time.time()

        train(TRAIN_DATA_PATH)

        end_time = time.time()
        duration = end_time - start_time

        mins, secs = divmod(duration, 60)
        logging.info(f"🕒 Total training time: {mins:.0f} minutes {secs:.2f} seconds")
        
    except Exception as e:
        logging.exception(f"Training pipeline failed due to: {e}")