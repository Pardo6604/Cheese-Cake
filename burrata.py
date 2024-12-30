import pandas as pd
import numpy as np
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
from keras import layers
import yfinance as yf

# Numerical:3
'''
Perceived performance / Competitiveness
Quality
Capital
Product Diversification
'''

#Categorical:
'''
Technological surveillance
Dynamic capability
Intellectual property
Employees skills
Facilities adequation
'''

def get_recommendationMean(df:pd.DataFrame):
    '''
    Calculates the mean from recomendations firms the scale is between -1 and 1.
    hold values are not being consider neither in the sum nor the total.

    Improvements: Consider the stock behaviour, for example a stock that passes 
    from sell to buy would have a higher value than others.
    '''

    strong_sell_val = -1*(df['strongSell'].sum())
    sell_val = -1/2*(df['sell'].sum())
    hold_val = 0*(df['hold'].sum())
    buy_val = 1/2*(df['buy'].sum())
    strong_buy_val = 1*(df['strongBuy'].sum())

    sum = strong_sell_val+sell_val+hold_val+buy_val+strong_buy_val
    total = abs(strong_sell_val) + abs(sell_val) + abs(hold_val) + abs(buy_val) + abs(strong_buy_val)

    return sum/total


def get_stats(ticker:str) -> np.array:
    company = yf.Ticker(ticker)
    
    recommendations = company.recommendations_summary
    recommendationMean = get_recommendationMean(recommendations) # Complete

    retrunOnEquity = company.info.get('retrunOnEquity')
    heldPercentInstitutions = company.info.get('heldPercentInstitutions')
    targetMeanPrice = company.info.get('targetMeanPrice')
    currentPrice = company.info.get('currentPrice')

    block = np.array([ticker, recommendationMean, retrunOnEquity, heldPercentInstitutions, targetMeanPrice, currentPrice])

    return block



def buildDf(ticker_list:list[str]):
    blockList = []

    for ticker in ticker_list:
        block = get_stats(ticker)
        blockList.append(block)

    blockList = np.vstack(blockList)
        
    blockDf = pd.DataFrame(blockList, columns=['ticker', 'recommendationMean', 'returnOnEquity', 'heldPercentInstitutions', 'targetMeanPrice', 'currentPrice'])

    blockDf['targetMeanPrice'] = blockDf['targetMeanPrice'] - blockDf['currentPrice']
    blockDf.rename(columns={'targetMeanPrice':'target - current'}, inplace = True)

    blockDf.drop(columns=['currentPrice'], inplace = True)

    return blockDf


#TESTS
test = buildDf(['AAPL','INTC'])
print(test)



"""
Model wiht 4 numerical inputs / 5 units, tanh activation / 6 units, relu activation / 2, tanh activation
"""
def model(dimensions:int, layer1:tuple[int, str], layer2:tuple[int, str], output:tuple[int, str]):

    inputs = keras.Input(shape = (dimensions,))
    
    dense = layers.Dense(layer1[0], activation=layer1[1])
    x = dense(inputs)

    x = layers.Dense(layer2[0], activation=layer2[1])(x)
    outputs = layers.Dense(output[0], activation=output[1])(x)

    model = keras.Model(inputs, outputs, name = 'Burrata')

