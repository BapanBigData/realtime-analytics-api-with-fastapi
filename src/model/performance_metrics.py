from sklearn.metrics import f1_score


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    f1_minority = f1_score(y_test, preds, pos_label=1)
    return f1_minority
