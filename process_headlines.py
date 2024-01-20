import pandas as pd
import re
import os


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    return url_pattern.sub('', text) if isinstance(text, str) else text


def preprocess_headlines(date_column,headline_column, data_folder):
    # specify folder name where headlines are
    headlines_path = os.path.join(data_folder, "Headlines")
    get_directory = os.path.join(headlines_path, "csv")
    save = "Processed headlines"
    if not os.path.exists(os.path.join(headlines_path, save)):
        os.makedirs(os.path.join(headlines_path, save))
    save_directory = os.path.join(headlines_path, save)

    headlines_df = []
    
    for headline_name in os.listdir(get_directory):
        specific_headlines = os.path.join(save_directory, f"{headline_name}")
        get_specific = os.path.join(get_directory, f"{headline_name}")
        if not os.path.exists(specific_headlines) and headline_name.endswith(".csv"):
            headline = pd.read_csv(get_specific, sep=",", encoding="ISO-8859-1", header=0, on_bad_lines='warn', engine='python')
            
            cleaned_headline = headline[[date_column, headline_column]].copy()
            cleaned_headline[date_column] = pd.to_datetime(cleaned_headline[date_column], errors='coerce')
            cleaned_headline[date_column] = cleaned_headline[date_column].ffill().dt.strftime("%Y-%m-%d")
            cleaned_headline = cleaned_headline.dropna(subset=[date_column, headline_column])
            cleaned_headline[headline_column] = cleaned_headline[headline_column].apply(remove_urls)
            cleaned_headline = cleaned_headline.drop_duplicates(subset=[headline_column])
            cleaned_headline.rename(columns={date_column: 'Date', headline_column: 'Headline'}, inplace=True)

            cleaned_headline.to_csv(specific_headlines, index=False)
            headlines_df.append(cleaned_headline)

            print(f"{headline_name} processed headline saved to csv")
        elif os.path.exists(specific_headlines):
            headlines = pd.read_csv(specific_headlines, parse_dates=["Date"])
            headlines_df.append(headlines)
            
    all_headlines_df = pd.concat(headlines_df, ignore_index=True)

    print("Processing headlines done")

    return all_headlines_df
