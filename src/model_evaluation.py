from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate
import pandas as pd

#Return the evaluation of the models without cross-validation
def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    """
    Fits and evaluates given ML models.
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data

    """
    #Create empty scores dictionary for each model
    model_scores = {}

    for name,model in models.items():

        #Fit model
        model.fit(X_train, y_train)

        #Predict on the test set
        y_pred = model.predict(X_test)

        #Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        scores = {
            "MSE": mse,
            "MAE": mae,
            "R^2": r2
        }

        model_scores[name] = scores

    return model_scores

def cross_validate(models, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]

    cv_results = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        cv_results[name] = {
            "MSE": -scores["test_neg_mean_squared_error"].mean(),
            "MSE_std": scores["test_neg_mean_squared_error"].std(),
            "MAE": -scores["test_neg_mean_absolute_error"].mean(),
            "R2": scores["test_r2"].mean(),
            "R2_std": scores["test_r2"].std()
        }

    return cv_results