import numpy as np
import os
from statsmodels.api import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

directory_path = "Replace with your directory path"
file_names = os.listdir(directory_path)

def portfolio(tickers, ):
    def create_portfolio_for_period(file_names, period):
        returns = []
    # stock_dates = []
        for ticker in tickers:
            df = pd.read_csv(os.path.join(directory_path, ticker))
            if 'Predicted' in df.columns and period < len(df):
                predicted_return = df.iloc[period]['Predicted']
                if not np.isnan(predicted_return):
                    returns.append((ticker.replace('.csv', ''), predicted_return))
        
                #stock_dates = stock_dates.append(df["Date"])

        if not returns:
            return 0

        sorted_returns = sorted(returns, key=lambda x: x[1], reverse=True)
        top_6 = sorted_returns[:6]
        bottom_6 = sorted_returns[-6:]
        portfolio_return = sum([x[1] for x in top_6]) / 6 - sum([x[1] for x in bottom_6]) / 6
        return portfolio_return

    file_path = "ABB_Analysis_Results.csv"
    stock_dates_read = pd.read_csv(os.path.join(directory_path, file_path))
    stock_dates = stock_dates_read["Date"]
    num_periods = len(pd.read_csv(os.path.join(directory_path, file_names[0])))
    portfolio_returns = [create_portfolio_for_period(file_names, period) for period in range(num_periods)]

    market_returns = "PATH RO MARKET RETURNS" 
    risk_free_rates = "PATH TO RISK FREE RATES"
    market_returns["Date"] = pd.to_datetime(market_returns["Date"], format="%Y-%m-%d")
    market_returns["Date"] = market_returns["Date"] - pd.DateOffset(days=1)
    risk_free_rates["Date"] = pd.to_datetime(risk_free_rates["Date"], format="%Y-%m-%d") 
    risk_free_rates["Risk_free"] = risk_free_rates["Risk_free"].str.replace('%', '').astype(float) / 100.0

    market_returns["Monthly_Return"] = market_returns["Close"].pct_change()
    market_returns.drop(["Open", "High", "Low", "Adj Close"], axis="columns", inplace=True)

    marged_data = pd.merge(market_returns, risk_free_rates, on="Date", how="inner")
    
    market_returns = market_returns.replace([np.inf, -np.inf], np.nan).dropna(subset=['Monthly_Return'])
    num_periods = min(len(marged_data), len(portfolio_returns))
    portfolio_returns = portfolio_returns[:num_periods]
    marged_data = marged_data[:num_periods]
    portfolio_returns_df = pd.DataFrame({"Date": stock_dates, "Monthly_Return": portfolio_returns})
    portfolio_returns_df['Date'] = pd.to_datetime(portfolio_returns_df['Date'])

    # CAPM Analysis
    market_returns = market_returns.loc[136:180]
    X = sm.add_constant(market_returns['Monthly_Return'])
    model = OLS(portfolio_returns, X).fit()
    portfolio_beta = model.params[1]

    risk_free_rates_series = pd.DataFrame({"Date": marged_data["Date"], "Risk_free":marged_data["Risk_free"]})

    # Regression Analysis
    regression_model = OLS(portfolio_returns, X).fit()

    alpha_capm = regression_model.params['const']
    beta_capm = portfolio_beta 
    p_value_alpha_capm = regression_model.pvalues['const']
    r_squared_capm = regression_model.rsquared
    f_statistic_capm = regression_model.f_pvalue
    durbin_watson_capm = sm.stats.stattools.durbin_watson(regression_model.resid)

    metrics_capm = {
        'Model': 'CAPM',
        'Alpha': alpha_capm,
        'P-value of Alpha': p_value_alpha_capm,
        'Beta Coefficients': beta_capm,
        'R-squared': r_squared_capm,
        'F-statistic p-value': f_statistic_capm,
        'Durbin-Watson': durbin_watson_capm
    }
    print("CAPM", metrics_capm)

    print(regression_model.summary())

    portfolio_returns_df['Cumulative_Return'] = (1 + portfolio_returns_df['Monthly_Return']).cumprod() - 1
    portfolio_returns_df['Rolling_Return'] = portfolio_returns_df['Monthly_Return'].rolling(window=30).mean()
    portfolio_returns_df.to_csv(r"C:\Users\User\OneDrive - Lund University\Kandidatuppsats\portfplio.csv")

    return_series = portfolio_returns_df["Monthly_Return"]
    cumulative = (1 + return_series).cumprod()
    rolling_max = cumulative.cummax()
    portfolio_returns_df["Drawdown"] = (cumulative - rolling_max) / rolling_max

    ##### fama french 3 factor
    excess_return = marged_data["Monthly_Return"] - risk_free_rates_series["Risk_free"]

    mkt = pd.DataFrame({"Date":marged_data["Date"], "Excess_return": excess_return})

    smb = "PATH TO SMB"
    hml = "PATH TO HML" 


    print("port", portfolio_returns_df.columns, "rf", risk_free_rates_series.columns, "mkt", mkt.columns, "smb", smb.columns, "hml", hml.columns)

    data = pd.concat([portfolio_returns_df["Monthly_Return"], risk_free_rates_series["Risk_free"], mkt["Excess_return"], smb["smb"], hml["hml"]], axis=1)
    data.columns = ['Ri', 'Rf', 'Mkt-Rf', 'SMB', 'HML']
    data = sm.add_constant(data)
    missing_values = data.isna().sum()
    print(missing_values)

    model = sm.OLS(data['Ri'], data[['const', 'Mkt-Rf', 'SMB', 'HML']])
    results = model.fit()

    results_summary = results.summary()

    latex_table = results_summary.as_latex()

    with open('regression_summary.tex', 'w') as f:
        f.write(latex_table)        

    alpha_ff3 = results.params['const']
    beta_ff3 = results.params[['Mkt-Rf', 'SMB', 'HML']].to_dict() 
    p_value_alpha_ff3 = results.pvalues['const']
    r_squared_ff3 = results.rsquared
    f_statistic_ff3 = results.f_pvalue
    durbin_watson_ff3 = sm.stats.stattools.durbin_watson(results.resid)



        ###### carhart four factór
    umd = "PATH TO UMD"

    data_car = pd.concat([portfolio_returns_df["Monthly_Return"], risk_free_rates_series["Risk_free"], mkt["Excess_return"], smb["smb"], hml["hml"], umd["umd"]], axis=1)
    data_car.columns = ['Ri', 'Rf', 'Mkt-Rf', 'SMB', 'HML', "UMD"]
    data_car = sm.add_constant(data_car)

    model_car = sm.OLS(data_car['Ri'], data_car[['const', 'Mkt-Rf', 'SMB', 'HML', 'UMD']])
    results_car = model_car.fit()
    results_car_summary = results_car.summary()
    latex_table_car = results_car_summary.as_latex()

    with open("carhart_reg_summary", "w") as f:
        f.write(latex_table_car)

    alpha_carhart = results_car.params['const']
    beta_carhart = results_car.params[['Mkt-Rf', 'SMB', 'HML', 'UMD']].to_dict()
    p_value_alpha_carhart = results_car.pvalues['const']
    r_squared_carhart = results_car.rsquared
    f_statistic_carhart = results_car.f_pvalue
    durbin_watson_carhart = sm.stats.stattools.durbin_watson(results_car.resid)

    metrics_ff3 = {
        'Model': 'Fama French 3-Factor',
        'Alpha': alpha_ff3,
        'P-value of Alpha': p_value_alpha_ff3,
        'Beta Coefficients': beta_ff3,
        'R-squared': r_squared_ff3,
        'F-statistic p-value': f_statistic_ff3,
        'Durbin-Watson': durbin_watson_ff3
    }

    metrics_carhart = {
        'Model': 'Carhart 4-Factor',
        'Alpha': alpha_carhart,
        'P-value of Alpha': p_value_alpha_carhart,
        'Beta Coefficients': beta_carhart,
        'R-squared': r_squared_carhart,
        'F-statistic p-value': f_statistic_carhart,
        'Durbin-Watson': durbin_watson_carhart
    }

    df_ff3 = pd.DataFrame([metrics_ff3])
    df_carhart = pd.DataFrame([metrics_carhart])

    combined_df = pd.concat([df_ff3, df_carhart], ignore_index=True)
    transposed_combined_df = combined_df.set_index('Model').T

    latex_transposed_table = transposed_combined_df.to_latex()

    with open('transposed_regression_summary.tex', 'w') as f:
        f.write(latex_transposed_table)

    print(latex_transposed_table) 