# Use a pipeline as a high-level helper
import tensorflow as tf
from transformers import AutoTokenizer, pipeline, TFAutoModelForSequenceClassification
import pandas as pd
import finnhub
import ast

finnhub_client = finnhub.Client(api_key="ctn0dbpr01qjlgir4mpgctn0dbpr01qjlgir4mq0")

# Model initialization (personalize) 
tokenizer = AutoTokenizer.from_pretrained("fuchenru/Trading-Hero-LLM")
pipe = pipeline("text-classification", model="fuchenru/Trading-Hero-LLM")
model = TFAutoModelForSequenceClassification.from_pretrained("fuchenru/Trading-Hero-LLM")

def get_news(ticker_list:list[str], start:str, end:str):
    '''
    Iterates through a list of tickers and retreive the news for those companies in the detailed preiod
    returns a dictionary of dataframes with all the news' information.
    '''
    ticker_news_dict = {}
    for ticker in ticker_list:
        news = finnhub_client.company_news(ticker, _from=start, to=end)
        news = pd.DataFrame(news)
        ticker_news_dict[ticker] = news
    
    return ticker_news_dict



def get_ratings(ticker_news:dict):
    '''
    Iterates through a data frame that contains all the news' information and uses the text clasification
    model to give the ticker an score and avarage them.
    '''
    ticker_rating_dict = {}

    for ticker, news in ticker_news.items():
        for index in range(len(news)):
            summary = news['summary'][index]

            inputs = tokenizer(summary, max_length=512, truncation=True, padding='max_length', return_tensors='tf')
            result = model(**inputs)
            result = tf.argmax(result.logits, axis=1).numpy().item()

            print(result)

            #Changing the numerical "Negative" representation to -1
            if result == 2:
                result = -1
            
            print(f"Score obtained per summary: Summary: {summary} - Result: {result}")

            try:
                ticker_rating_dict[ticker] += result

            except KeyError:
                ticker_rating_dict[ticker] = result
            
        ticker_rating_dict[ticker] /= len(news)
    
    return ticker_rating_dict

def ripen(ticker_list:list[str], start:str, end:str):
    '''
    Bind the other two previous functions
    '''
    ticker_news_dict = get_news(ticker_list, start, end)
    ticker_rating_dict = get_ratings(ticker_news_dict)

    return ticker_rating_dict


         










# News API key: ctn0dbpr01qjlgir4mpgctn0dbpr01qjlgir4mq0

#Apple Test

'''
res = finnhub_client.company_news('AAPL', _from="2024-12-25", to="2024-12-27")
res = pd.DataFrame(res)

print(len(res)) 

'''

'''
pipe = pipeline("text-classification", model="fuchenru/Trading-Hero-LLM")
result = pipe("Investor sentiment improved following news of a potential trade deal")

print(type(result))
for x in result:
    print(x)
'''


