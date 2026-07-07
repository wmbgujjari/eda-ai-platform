def classical_inference(model, train_df, features, target_col):
    """
    Predict using LGBM / classical ML model.
    """
    y_true = train_df[target_col]
    y_pred = model.predict(train_df[features])
    return y_true, y_pred
