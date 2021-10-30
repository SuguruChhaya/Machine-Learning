import pandas as pd
import quandl

#I guess quandl is a module that helps me pull data from the web. 
#!The get won't work if I am not connected to internet. 
df = quandl.get('WIKI/GOOGL')

#Wanna simplify data as much as possible and use as many meaningful features as possible. So take away the necessary parts. 
#I am guessing that the df data structure like a dictionary. 
#Some talk about feature or label? What is the difference?
#Label represents data? Kind of confused...
#print(df.head())

df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#Just following through this, I understand how the Adj. Close is the answer value and we feed couple of question and answer. 

#Don't wanna delete data so fillna? 

forecase_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecase_out = int(math.ceil(0.1*len(df)))
#Forcast out is where I set the number of days I wanna predict out. So if I get data for 10 days I am predicting 1 day after that I guess? 

#Yeah, I'll need to learn some data science to actually understand this shit. 