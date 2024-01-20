import pandas as pd
import os

def get_monthly_prices(drop_columns, frequency, time_returns, price_close, data_folder):

    stock_price_path = os.path.join(data_folder, "Aktiepriser t.o.m 2023-12-10")
    stock_dfs = []
    save = f"{frequency} Prices"
    if not os.path.exists(os.path.join(stock_price_path, save)):
        os.makedirs(os.path.join(stock_price_path, save))
    save_directory = os.path.join(stock_price_path, save)
    
    
 
    for stock_name in os.listdir(stock_price_path):
        specific_stock = os.path.join(save_directory, f"{stock_name}")
        if not os.path.exists(specific_stock) and stock_name.endswith(".csv"):
            stock = pd.read_csv(os.path.join(stock_price_path, f"{stock_name}"), sep=",", encoding="ISO-8859-1", header=0, parse_dates=["Date"], index_col="Date")
            if all(column in stock.columns for column in drop_columns):
                stock.drop(drop_columns, axis="columns", inplace=True)
            
            stock = stock.resample(frequency).last()
            stock[time_returns] = stock[price_close].pct_change(fill_method=None)

            stock.to_csv(specific_stock)
            stock_dfs.append(stock)

            print(f"{stock_name} processed stock prices saved to csv")

        elif os.path.exists(specific_stock):
            stock = pd.read_csv(specific_stock, parse_dates=["Date"], index_col="Date")
            stock_dfs.append(stock)


    all_stocks_df = pd.concat(stock_dfs, ignore_index=False)
    
    print("Processing stock prices done")

    return all_stocks_df
