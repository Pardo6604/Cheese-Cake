from burrata import buildDf
from gouda import ripen
import pandas as pd
import numpy as np


ticker_list = ['INTC', 'AAPL']
news_scores_dict = ripen(ticker_list, '2024-12-25', '2024-12-27')
stats_df = buildDf(ticker_list)

news_scores_series = pd.Series(news_scores_dict, name = 'news_scores_series') # Transforming the dictionary into a pandas series
stats_df = stats_df.set_index('ticker').join(news_scores_series) # Appending news scores from the dictionary to the dataframe

print(stats_df)