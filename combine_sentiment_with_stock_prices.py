import pandas as pd
import numpy as np
import os

def get_combined_data(combined_stock_sent):
    
    combineddata_read = pd.read_csv(combined_stock_sent, sep=",", encoding="ISO-8859-1")
    print("Processing combining headlines and stocks done")
    return combineddata_read

def load_all_sentiment_data(tickers, thesentimentpath, start_date_str, end_date_str):
    all_sentiment_data = []

    for ticker in tickers:
        thesentimentpath = thesentimentpath.format(ticker=ticker, start_date_str=start_date_str, end_date_str=end_date_str)
        if os.path.exists(thesentimentpath):
            sentiment_df = pd.read_csv(thesentimentpath)
            sentiment_df["Asset"] = ticker
            all_sentiment_data.append(sentiment_df)

    return pd.concat(all_sentiment_data, ignore_index=True)

def load_all_stocks(tickers, specific_stock_path):
    all_stocks = []
    print("stockprice", specific_stock_path)
    for stock in tickers:
        if os.path.exists(specific_stock_path):
            stock_df = pd.read_csv(specific_stock_path)
            stock_df["Asset"] = stock
            all_stocks.append(stock_df)
            print("stockdf",stock_df.head())
            
            return stock_df
        

def combine_sent_stock_price(ticker, combined_stock_sent, tickers, thesentimentpath, start_date_str, end_date_str, frequency):
    data_folder = os.path.join(os.getcwd(), "Data")
    specific_stock_path = os.path.join(data_folder, "Aktiepriser t.o.m 2023-12-10", f"{frequency} Prices", ticker+".csv")

    all_sentiment_da = load_all_sentiment_data(tickers, thesentimentpath, start_date_str, end_date_str)
    all_sentiment_data = add_aggregated_sentiment_feature(all_sentiment_da, sentiment_column="Numeric Sentiment", date_column='Date')
    all_sentiment_data["Date"] = pd.to_datetime(all_sentiment_data["Date"])
    current_asset_sentiment = all_sentiment_data[all_sentiment_data["Asset"] == ticker]
    current_asset_sentiment["Date"] = pd.to_datetime(current_asset_sentiment["Date"])
    current_asset_sentiment.set_index('Date', inplace=True)

    resampled_sentiment = current_asset_sentiment["Weighted_Sentiment"].resample(frequency).mean()


    stock_prices = load_all_stocks(tickers, specific_stock_path)    
    stock_prices['Date'] = pd.to_datetime(stock_prices['Date'])
    stock_prices.set_index("Date", inplace=True)

    resampled_stock_prices = stock_prices.resample(frequency).last()

    combined_data_all = pd.merge(resampled_stock_prices, resampled_sentiment, on="Date",  how="right")
    print(combined_data_all)

    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    
    combined_data_all = combined_data_all[(combined_data_all.index >= start_date) & (combined_data_all.index <= end_date)]
    combined_data_all.sort_index(inplace=True)
    combined_data_all["Daily Returns"] = combined_data_all["Close"].pct_change()
    combined_data_all["Asset"] = ticker

    combined_data_all.to_csv(combined_stock_sent, index=True)


    print(f"Combined data for {ticker} saved to csv at {combined_stock_sent}")

    return combined_data_all


def calculate_weighted_sentiment(df, sentiment_column='Numeric Sentiment', date_column='Date', decay_factor=0.95):
    weighted_sentiment = df.groupby(date_column).apply(lambda x: (x[sentiment_column] * np.power(decay_factor, np.arange(len(x))[::-1])).sum() / np.power(decay_factor, np.arange(len(x))[::-1]).sum())

    return weighted_sentiment

def add_aggregated_sentiment_feature(df, sentiment_column="Numeric Sentiment", date_column='Date'):
    weighted_sentiment = calculate_weighted_sentiment(df, sentiment_column, date_column)
    df = df.merge(weighted_sentiment.rename('Weighted_Sentiment'), left_on=date_column, right_index=True)
    
    return df
