from pycaret.regression import predict_model

def automl_inference(model, train_df, target_col):
    """
    Predict using PyCaret AutoML model.
    """
    preds_df = predict_model(model, data=train_df)
    y_true = train_df[target_col]
    y_pred = preds_df["prediction_label"]

    return y_true, y_pred
