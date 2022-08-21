from keras.models import load_model
import numpy as np
import pandas as pd
import pandas_datareader as data
import datetime as dt
from datetime import timedelta 
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit as st


model = load_model('model1.h5')

ticker = yf.Ticker('^GSPC')
data2=ticker.history(period='12mo')
data1=data2.tail(150)

user_input =st.text_input("Enter date in dd/mm/yyyy:")

st.write(data1.head())

df1=data1.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
date1 = user_input
date1=str(date1)
date1=dt.datetime.strptime(date1,"%d/%m/%Y")
nDay=date1-datemax
nDay=nDay.days
date_rng = pd.date_range(start=datemax, end=date1, freq='D')
date_rng=date_rng[1:date_rng.size]
lst_output=[]
n_steps=150
i=0
while(i<=nDay):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
res =scaler.inverse_transform(lst_output)
print(res)
output = res[nDay]

st.write("predicted price : ",output)

predictions=res[res.size-nDay:res.size]
print(predictions.shape)
predictions=predictions.ravel()
print(type(predictions))
print(date_rng)
print(predictions[1])
print(date_rng.shape)

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')
df = pd.DataFrame(data = date_rng)
df['Predictions'] = predictions.tolist()
df.columns =['Date','Price']
st.write(df)
csv = convert_df(df)
st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)
#visualization

fig =plt.figure(figsize=(10,6))
xpoints = date_rng
ypoints =predictions
plt.xticks(rotation = 90)
plt.plot(xpoints, ypoints)
st.pyplot(fig)

