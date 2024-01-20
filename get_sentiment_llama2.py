from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import re
from datetime import datetime


def load_model(model_path):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler])
    llama_model = LlamaCpp(
        model_path=model_path,
        temperature=0,
        max_tokens=512,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,
    )

    return llama_model

def initialize_langchain(llama_model, template, input_variables):
   prompt = PromptTemplate(template=template, input_variables=input_variables)
   llm_chain = LLMChain(prompt=prompt, llm=llama_model)

   return llm_chain

def extract_valid_sentiment(sentiment_result):
    valid_labels = r"\b(Buy|Hold|Sell)\b"
    match = re.search(valid_labels, sentiment_result, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


########## analys
def analyze_sentiment_batch(filtered_headlines, llm_chain, ticker, max_retries=10):
    sentiments = []
  
   
    for _, row in filtered_headlines.iterrows():
        headline = row["Headline"]
        headline_date = row["Date"]
    
        extracted_sentiment = None
        attempts=0

        while extracted_sentiment is None and attempts < max_retries:
            input_data = {"text": headline, "ticker": ticker}
            sentiment_result = llm_chain.run(input_data)
            extracted_sentiment = extract_valid_sentiment(sentiment_result)

            attempts += 1
    

        if extracted_sentiment is None:
            extracted_sentiment = "neutral"  

        sentiments.append(extracted_sentiment)
        print("Company: ", ticker, "Date:", headline_date, "Sentiment of", headline, "is", extracted_sentiment)
        
    return sentiments

def analyze_and_aggregate_sentiment(processed_headlines, llm_chain, ticker, start_date_str, end_date_str,thesentimentpath):
    sentiment_to_numeric = {"Sell": -1, "Hold": 0, "Buy": 1}

    filtered_headlines = processed_headlines
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    filtered_headlines["Date"] = pd.to_datetime(filtered_headlines["Date"], errors="coerce")
    
    if start_date and end_date:
        filtered_headlines = filtered_headlines[(filtered_headlines["Date"] >= start_date) & (filtered_headlines["Date"] <= end_date)]
 
    filtered_headlines["Sentiment"] = analyze_sentiment_batch(filtered_headlines, llm_chain, ticker) 
    filtered_headlines["Numeric Sentiment"] = filtered_headlines["Sentiment"].map(sentiment_to_numeric)
    filtered_headlines = filtered_headlines.dropna(subset=["Date"])

    filtered_headlines.to_csv(thesentimentpath)
    print("Sentimented headlines saved to output")

    return filtered_headlines

def sentimented_headlines(thesentimentpath):
    
    sentiment_read = pd.read_csv(thesentimentpath, sep=",", encoding="ISO-8859-1")
    print("Processing sentimented headlines done")

    return sentiment_read