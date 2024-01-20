# random forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint as sp_randint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os                                              
from sklearn.model_selection import TimeSeriesSplit
import time
import tikzplotlib



    
def create_features_wide(all_stocks_combined):
    all_stocks_combined["Asset"] = all_stocks_combined["Asset"].fillna(method="ffill")
    sentiment_wide = all_stocks_combined.pivot_table(index="Date", columns="Asset", values="Weighted_Sentiment", fill_value=0)
    close_wide = all_stocks_combined.pivot_table(index="Date", columns="Asset", values="Daily Returns", fill_value=0)
    features_wide = sentiment_wide.join(close_wide, lsuffix="_sentiment", rsuffix="_close")
    return features_wide
    


def process_all_tickers(tickers, all_stocks_combined, features_wide, output_folder, combined_stock_sent,processed_stocks):
    """
    Process each ticker through the Random Forest model.
    """
        
    for ticker in tickers:
        print("Random Forest")
        rf_results = random_forest(output_folder, ticker, features_wide, all_stocks_combined)
       

def random_forest(output_folder, ticker, features_wide, all_stocks_combined):
    
    ticker_returns = all_stocks_combined[all_stocks_combined["Asset"] == ticker][["Date", "Daily Returns"]].set_index("Date")
    ticker_features = features_wide.join(ticker_returns)
    # for running without sentiment
    #sentiments_list = ["Daily Returns", 'ABB_sentiment', 'ALFA_sentiment', 'ALIV-SDB_sentiment', 'ASSA-B_sentiment', 'ATCO-A_sentiment', 'AZN_sentiment', 'BOL_sentiment', 'ELUX-B_sentiment', 'ERIC-B_sentiment', 'ESSITY-B_sentiment', 'EVO_sentiment', 'GETI-B_sentiment', 'HEXA-B_sentiment', 'HM-B_sentiment', 'INVE-B_sentiment', 'KINV-B_sentiment', 'NDA-SE_sentiment', 'NIBE-B_sentiment', 'SAND_sentiment', 'SBB-B_sentiment', 'SCA-B_sentiment', 'SEB-A_sentiment', 'SHB-A_sentiment', 'SINCH_sentiment', 'SKF-B_sentiment', 'SWED-A_sentiment', 'TEL2-B_sentiment', 'TELIA_sentiment', 'VOLV-B_sentiment']
    ticker_features = ticker_features.dropna(subset=["Daily Returns"])
    ticker_returns = ticker_returns["Daily Returns"].shift(-1)
    total_dates = len(all_stocks_combined)
    train_end = int(total_dates*0.4)
    val_end = train_end+int(total_dates*0.5)
    test_end = total_dates
    all_dates = all_stocks_combined["Date"].values
    X = ticker_features.drop(["Daily Returns"], axis=1)
    y = ticker_features["Daily Returns"]



    actuals = []
    predictions_list = []
    validation_scores = []
    dates_list = []
    n_splits = 5
    validation_size = 0.1  

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_index, test_index in tscv.split(X):
        X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

        val_split_index = int(len(X_train_full) * (1 - validation_size))
        X_train, X_val = X_train_full[:val_split_index], X_train_full[val_split_index:]
        y_train, y_val = y_train_full[:val_split_index], y_train_full[val_split_index:]

        rf = RandomForestRegressor(n_estimators=300, 
                                   bootstrap=True, 
                                   max_depth=10, 
                                   max_features="sqrt",
                                min_samples_leaf=4, 
                                min_samples_split=5, 
                                random_state=42)
        rf.fit(X_train, y_train)

        val_predictions = rf.predict(X_val)
        print("validate predic", val_predictions)
        val_mse = mean_squared_error(y_val, val_predictions)
        val_r2 = r2_score(y_val, val_predictions)
        validation_scores.append((val_mse, val_r2))
        print(f"Validation MSE: {val_mse}, Validation R^2: {val_r2}")

        test_predictions = rf.predict(X_test)
        predictions_list.extend(test_predictions)
        actuals.extend(y_test)
        dates_list.extend(X_test.index)

        test_mse = mean_squared_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        print(f"Test MSE: {test_mse}, Test R^2: {test_r2}")

    actuals_array = np.array(actuals)
    predictions_array = np.array(predictions_list)

    actuals_mean = np.mean(actuals_array)
    actuals_median = np.median(actuals_array)
    actuals_std_dev = np.std(actuals_array)

    predictions_mean = np.mean(predictions_array)
    predictions_median = np.median(predictions_array)
    predictions_std_dev = np.std(predictions_array)

    correlation = np.corrcoef(actuals_array, predictions_array)[0, 1]

    results_df = pd.DataFrame({
        "Date": dates_list,
        "Actual": actuals, 
        "Predicted": predictions_list
    })

    results_df.to_csv(output_folder, index = False)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")


    rf_results = {
        "Mean Squared Error": test_mse,
        "R^2 Score": test_r2,
        "Predictions" : predictions_list,
        "Importances" : importances,
        "CV-scores" : cv_scores,
        "MSE_validate" : val_mse,
        "R^2 Score_val" : val_r2,
        "Actuals Mean": actuals_mean,
        "Actuals Median": actuals_median,
        "Actuals Std Dev": actuals_std_dev,
        "Predictions Mean": predictions_mean,
        "Predictions Median": predictions_median,
        "Predictions Std Dev": predictions_std_dev,
        "Correlation": correlation
    }

    output = os.path.join(output_folder)

    with open(output, "w") as file:
        file.write("Testing:\nMean Squared Error: " + str(rf_results["Mean Squared Error"]) + "\n")
        file.write("R^2 Score: " + str(rf_results["R^2 Score"]) + "\n")
        file.write("Predictions: "+ str(rf_results["Predictions"])+"\n")
        file.write("Importances: "+ str(rf_results["Importances"])+"\n")
        file.write("CV-scores: " + str(rf_results["CV-scores"])+"\n\n")
        file.write("Validation:\nMSE_validate: " + str(rf_results["MSE_validate"])+"\n")
        file.write("R^2 Score: " + str(rf_results["R^2 Score_val"])+"\n")
        file.write("Statistical Measures:\n")
        file.write(f"Actuals - Mean: {rf_results['Actuals Mean']}, Actuals - Median: {rf_results['Actuals Median']}, Actuals - Std Dev: {rf_results['Actuals Std Dev']}\n")
        file.write(f"Predictions - Mean: {rf_results['Predictions Mean']}, Predictions - Median: {rf_results['Predictions Median']}, Predictions - Std Dev: {rf_results['Predictions Std Dev']}\n")
        file.write(f"Correlation between actuals and predictions: {rf_results['Correlation']}\n")

    return rf_results
