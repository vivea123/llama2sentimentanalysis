
import json
import os
import time
import platform
from process_stock_prices import get_monthly_prices
from process_headlines import preprocess_headlines
from get_sentiment_llama2 import load_model, initialize_langchain, analyze_and_aggregate_sentiment, sentimented_headlines
from combine_sentiment_with_stock_prices import combine_sent_stock_price, get_combined_data
from random_forest import process_all_tickers, create_features_wide


### specify variables
model_path = "PATH TO MODEL PATH"
jsonpath = "PATH TO JSON FILE"
start_date_str = "SPECIFY START DATE FOR ANALYSIS (yyyy-mm-dd)"
end_date_str = "SPECIFY END DATE FOR ANALYSIS (yyyy-mm-dd)"

price_close = "Close"
drop_columns = ["Open", "High", "Low", "Adj Close"] # drop columns in stock prices csv

date_column = "SPECIFY DATE COLUMN NAME"
headline_column = "SPECIFY HEADLINE COLUMN NAME"

frequency = "D" # "M"onthly, "W"eekly, "D"aily

###

with open(jsonpath, "r") as file:
        data = json.load(file)
        tickers = data["tickers"]

time_returns = "Monthly Returns" 

work_dir = os.getcwd()
data_folder = os.path.join(work_dir, "Data")
output_folder = os.path.join(work_dir, "Output")

if not os.path.exists(output_folder): 
    os.makedirs(output_folder)




for ticker in tickers:
    print(f"Analyzing {ticker}")
    combined_stock_sent = os.path.join(output_folder, "Combined Data", f"{ticker}_combined_data.csv")
    thesentimentpath = os.path.join(output_folder, "Sentiment", f"Sentiment_of_{ticker}_{start_date_str}_to_{end_date_str}.csv")
    
    def main():
        template = """
        [INST] <<SYS>>
            Act as an expert in stock trading holding {ticker}. 
            Answer in one token: positive for buy, negative for sell, or neutral for hold position.
        <</SYS>>
            Based only on the headline '{text}', will you buy, sell or hold {ticker} in the short term?
        [/INST]
        """

        input_variables = ["text", "ticker"]

        # 1. Initialize the model
        llama_model = load_model(model_path)
        llm_chain = initialize_langchain(llama_model, template, input_variables)

        # 2. Process all stock prices
        processed_stocks = get_monthly_prices(drop_columns, frequency, time_returns, price_close, data_folder)    
        
        # 3. Process all headlines
        processed_headlines = preprocess_headlines(date_column,headline_column, data_folder)
       
        # 4. Llama sentiment analysis on the headlines
        if os.path.exists(thesentimentpath):
            get_sentiment = sentimented_headlines(thesentimentpath)   
        elif not os.path.exists(thesentimentpath):
            get_sentiment = analyze_and_aggregate_sentiment(processed_headlines, llm_chain,ticker, start_date_str, end_date_str, thesentimentpath) 

        # 5. Combine sentiment with the stock return
        if not os.path.exists(combined_stock_sent):
            combine_analysis = combine_sent_stock_price(ticker, combined_stock_sent, tickers, thesentimentpath, start_date_str, end_date_str, frequency)
        elif os.path.exists(combined_stock_sent):
            get_combined = get_combined_data(combined_stock_sent)
            combine_analysis = get_combined

        # 6. Run random forest
        if os.path.exists(combined_stock_sent):
            #all_stocks_combined = combine_all_tickers(tickers, combined_stock_sent, output_folder)
            features_wide = create_features_wide(combine_analysis)
            results = process_all_tickers(tickers, combine_analysis, features_wide, output_folder, combined_stock_sent, processed_stocks)

           
        elif not os.path.exists(combined_stock_sent):
            print("Du har inte sentimentet för denna aktien")
            time.sleep(5)

        # # 7. Build a portfolio and evaluate against CAPM, F3FF, C4F
            

        analyserade_aktier = []
        analyserade_aktier.append(ticker) 

        print(f"Analys klar för {ticker}\nAlla tickers analyserade so far: ",analyserade_aktier)
        time.sleep(1)

    if __name__ == "__main__":
        main()