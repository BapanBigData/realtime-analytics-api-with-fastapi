from imblearn.over_sampling import SMOTE
from src.config.config import RANDOM_SEED


def apply_smote(X_train, y_train, random_state=RANDOM_SEED):
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    return X_train_smote, y_train_smote
