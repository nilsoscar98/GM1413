### Importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
import pickle
import spacy
from collections import Counter
import yfinance as yf
### Declaring functions
#Function that reads all the pickle files in a folder, returns list of file names.
def read_file_names(folder_path: str) -> list[str]:
    files = [f for f in os.listdir(folder_path) if f.endswith('.p')]
    
    return files
def extract_data(folder_path: str, ticker: str) -> list[dict]:
    file_path = os.path.join(folder_path, ticker)  #File path.

    with open(file_path, 'rb') as file:  #Open file.
        content = pickle.load(file)  #Load content.
        data = []  #Create empty list.

        for entry in content:  #Iterate over the content.
            data.append(entry)  #Collect the dictionary.

    return data  #Return the list of dictionaries.
# Putting all in a function so that we can use it later 
def process_text(text):
    nlp = spacy.load('en_core_web_md') #Load the spaCy model.
    words=[word.lemma_.lower() for word in nlp(text) if not (word.is_space or word.is_stop or word.is_punct)]
    return words
def extract_and_process_transcripts(transcripts: list[dict]) -> list[list[str]]:
    transcripts_texts = []
    times = []

    for t in transcripts:
        text = ''
        for speech in t['transcript']:
            text = str(text + ' '+ speech['speech'][0])
            
        transcripts_texts.append(text)
        times.append(t['time'])

    return transcripts_texts, times
def define_LM(LM_file:str) -> tuple:
    LM_negative=pd.read_excel(LM_file, sheet_name='Negative', header=None)
    LM_positive=pd.read_excel(LM_file, sheet_name='Positive', header=None)
    LM_uncertainty=pd.read_excel(LM_file, sheet_name='Uncertainty', header=None)

    LM_positive_list = LM_positive.squeeze().tolist()  # Squeeze is used to flatten the DataFrame column
    LM_negative_list = LM_negative.squeeze().tolist()
    LM_uncertainty_list = LM_uncertainty.squeeze().tolist()

    LM_positive_set = set(process_text(' '.join(LM_positive_list)))
    LM_negative_set = set(process_text(' '.join(LM_negative_list)))
    LM_uncertainty_set = set(process_text(' '.join(LM_uncertainty_list)))
    
    return LM_positive_set, LM_negative_set, LM_uncertainty_set
def score_transcript(text):
    if len(text) > 20_000:
        text = text[:20_000]
    
    words = process_text(text)
    number_of_words = len(words)
    counts = Counter(words)
    keys = set(counts.keys())

    pos = round((sum([counts[k] for k in (keys & LM_positive_set)]) / number_of_words), 4)
    neg = round((sum([counts[k] for k in (keys & LM_negative_set)]) / number_of_words), 4)
    unc = round((sum([counts[k] for k in (keys & LM_uncertainty_set)]) / number_of_words), 4)
    
    return (pos, neg, unc)

from concurrent.futures import ProcessPoolExecutor

def analysis(in_data: list[dict], ticker_names: list[str]) -> pd.DataFrame:
    df_all = pd.DataFrame(columns=['date', 'ticker', 'net_sentiment', 'pos', 'neg', 'unc', 'adj_close', 'returns'], dtype=object)

    # Step 1: Extract all tickers and their respective start dates
    start_dates = []
    for company in in_data:
        _, times = extract_and_process_transcripts(company)
        start_date = (pd.to_datetime(times[-1]) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        start_dates.append(start_date)
    
    # Step 2: Download market data for all tickers in one go
    market_data = yf.download(ticker_names, start=min(start_dates))
    
    # Step 3: Process each company in parallel
    def process_company(tic, company):
        transcripts, times = extract_and_process_transcripts(company)
        ticker = ticker_names[tic]
        start_date = start_dates[tic]
        company_data = market_data[market_data.index >= start_date]
        company_data['returns'] = company_data['Adj Close'] - company_data['Adj Close'].shift(1)
        # Further processing and appending to df_all
        # ...

    with ProcessPoolExecutor() as executor:
        executor.map(process_company, range(len(in_data)), in_data)
    
    return df_all
def group_by_sector(dataframe:pd.DataFrame, sector:str, sector_groups:list[str]) -> pd.DataFrame:
    df_sector = dataframe[dataframe['ticker'].isin(sector_groups[sector])]

    return df_sector
### Main function
if __name__ == "__main__":
    folder_path = 'tickers' #Path to the folder containing the pickle files.
    file_names = read_file_names(folder_path) #List of all the pickle files in the folder.
    tickers = [ticker[:-2] for ticker in file_names] #List of all the tickers.
    in_data = [] #Initialise an empty list to store the transcripts.

    #Loop through all files and extract transcripts.
    for file in file_names:
        transcripts = extract_data(folder_path, file) #Extract the transcripts.
        in_data.append(transcripts) #Append the transcripts to the list.

    LM_file = 'LoughranMcDonald_SentimentWordLists_2018.xlsx'
    LM_positive_set, LM_negative_set, LM_uncertainty_set = define_LM(LM_file)

    df_tickers = analysis(in_data, tickers)

    sector_groups = pd.DataFrame({
        'communication_services': ['CHTR', 'CMCSA', 'DIS', None, None, None, None, None, None],
        'consumer_cyclical':      ['BKNG', 'LOW', 'MCD', 'NKE', 'SE', None, None, None, None],
        'consumer_defensive':     ['PEP', 'PM', 'TGT', None, None, None, None, None, None],
        'energy':                 ['CVX', 'XOM', None, None, None, None, None, None, None],
        'financial_services':     ['AXP', 'BX', 'C', 'JPM', 'MA', 'MCO', 'TD', None, None],
        'healthcare':             ['ABBV', 'ABT', 'AZN', 'CVS', 'LLY', 'NVO', 'SNY', None, None],
        'industrials':            ['BA', 'HON', 'UNP', 'UPS', None, None, None, None, None],
        'technology':             ['AAPL', 'AMD', 'CRM', 'IBM', 'INTC', 'MSFT', 'ORCL', 'SAP', 'TXN']
    })
