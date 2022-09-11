import pandas as pd
import requests
import json
from matplotlib import style
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
import config

Dependent_Variable = 'AIG'
Independent_Variable = 'MET'

class core_stock():
    def daily(symbol, function = 'TIME_SERIES_DAILY', outputsize='full', api_key = config.alpha_vantage_api_key):
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        return data
 
IV = core_stock.daily(Independent_Variable)
DV = core_stock.daily(Dependent_Variable)

IV = pd.DataFrame.from_dict(IV['Time Series (Daily)'], orient='index')
IV[f'{Independent_Variable}_close'] = IV['4. close'].astype('float')
IV = IV.drop(['1. open', '2. high', '3. low', '4. close', '5. volume'], axis=1)

DV = pd.DataFrame.from_dict(DV['Time Series (Daily)'], orient='index')
DV[f'{Dependent_Variable}_close'] = DV['4. close'].astype('float')
DV = DV.drop(['1. open', '2. high', '3. low', '4. close', '5. volume'], axis=1)

df = pd.concat([IV,DV],join='inner', axis=1)
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['index'])
df = df[df['Date'].dt.year > 2021]

x = np.c_[df[f'{Independent_Variable}_close']]
y = np.c_[df[f'{Dependent_Variable}_close']]
model = linear_model.LinearRegression().fit(x,y)

prediction = model.predict(x)
residual = (y - prediction)

style.use('seaborn')
plt.figure(figsize=(20,10))
plt.title("Regression Plot")
plt.xlabel(f'{Independent_Variable}')
plt.ylabel(f'{Dependent_Variable}')
plt.scatter(x,y,color="black")
plt.plot(x,prediction,color="blue",linewidth=3)
plt.savefig(f'{Independent_Variable} - {Dependent_Variable} (Regression).png', format='png',dpi=300)

plt.figure(figsize=(20,10))
sns.residplot(data=df,x=prediction,y=residual,line_kws={"color": "green"},scatter_kws={"color": "black"})
plt.title('Residual Plot')
plt.ylabel('Residual Values')
plt.savefig(f'{Independent_Variable} - {Dependent_Variable} (Residuals).png', format='png',dpi=300)

print(f'{Dependent_Variable} is typically',round(np.mean(x-y),2), f'below {Independent_Variable}\n')
print("R squared = ",model.score(x,y))
print("Intercept = ",model.intercept_[0])
print("Coefficient = ",model.coef_[0][0])
print("Mean Absolute Error = ",metrics.mean_absolute_error(x,y))
print("Mean Squared Error = ",metrics.mean_squared_error(x,y))