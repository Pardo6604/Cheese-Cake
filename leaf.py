from burrata import buildDf
from gouda import ripen
import pandas as pd
import numpy as np


SP_list = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "BRK.B", "META", "JPM", "UNH",
    "PG", "HD", "V", "DIS", "XOM",
    "PFE", "KO", "NFLX", "PEP", "CSCO",
    "ADBE", "INTC", "BAC", "T", "CMCSA"
]

tsx_60_tickers = [
    "AEM", "BNS", "CP", "SHOP", "TD", "RY", "ENB", "CNR", "BMO", "WCN", 
    "SU", "FTS", "TRP", "MFC", "T", "BAM", "POT", "GIB", "WEED", "L", 
    "ATD", "MG", "NA", "BCE", "MKP", "X", "AC"
]

small_med_tickers = [
    "PLUG", "FIZZ", "ENPH", "RUN", "BEAM", "TWLO", "CRSP", "PINS", "SHOP", "ZM",
    "AVGO", "SQ", "LYFT", "RKT", "UPST", "WIX", "TTD", "DOCU", "FSLY", "ETSY",
    "MELI", "TLRY", "CGC", "CRWD", "DKNG"
]

small_mid_canadian_tickers = [
    "BFS", "ECN", "CTT", "LIF", "CGX", "PHO", "AUR", "TLM", "CUM", "MDA",
    "NAC", "DIR.UN", "DND", "VCI", "CST", "NHC", "XBC", "AIF", "DYN", "LXX",
    "TGN", "FF", "ISV", "KPT", "ROXG", "PEY", "QEC"
]

start = '2025-01-03'
end = '2025-01-05'


def wrap(ticker_list:list[str], start:str, end:str):
    # Using Gouda and Burrata to stract data
    news_scores_dict = ripen(ticker_list, start, end)
    stats_df = buildDf(ticker_list)
    
    # Binding Outputs
    news_scores_series = pd.Series(news_scores_dict, name = 'news_scores_series') # Transforming the dictionary into a pandas series
    stats_df = stats_df.set_index('ticker').join(news_scores_series) # Appending news scores from the dictionary to the dataframe
    
    return stats_df



stats_df = wrap(['AAPL'], start, end)
print(stats_df)

#stats_df.to_pickle('S&P_list.pkl')





