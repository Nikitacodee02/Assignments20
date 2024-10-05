#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[12]:


# Load the dataset without parsing dates

df = pd.read_csv('exchange_rate.csv', parse_dates=['date'], index_col='date')
df
# Display the first few rows and the column names
print(df.head())
print(df.columns)


# In[13]:


plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Ex_rate'], label='Exchange Rate')
plt.title('USD to Australian Dollar Exchange Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()


# In[4]:


# Check for missing values
print(df.isnull().sum())

# Fill or interpolate missing values
df['Ex_rate'].interpolate(method='time', inplace=True)

# Confirm there are no more missing values
print(df.isnull().sum())


# In[5]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# ACF and PACF plots
plot_acf(df['Ex_rate'], lags=20)
plt.show()

plot_pacf(df['Ex_rate'], lags=20)
plt.show()


# In[6]:


# Fit the ARIMA model (example with parameters p=1, d=1, q=1)
arima_model = ARIMA(df['Ex_rate'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Print model summary
print(arima_result.summary())

# Forecasting
forecast = arima_result.forecast(steps=10)
print(forecast)


# In[7]:


# Plotting the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index[-50:], df['Ex_rate'][-50:], label='Actual')
plt.plot(pd.date_range(df.index[-1], periods=10, freq='D'), forecast, label='Forecast')
plt.title('ARIMA Model - Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()


# In[8]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit the Exponential Smoothing model (Holt-Winters method as an example)
exp_model = ExponentialSmoothing(df['Ex_rate'], trend='add', seasonal=None, seasonal_periods=12)
exp_result = exp_model.fit()

# Forecasting
exp_forecast = exp_result.forecast(steps=10)
print(exp_forecast)

# Plotting the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index[-50:], df['Ex_rate'][-50:], label='Actual')
plt.plot(pd.date_range(df.index[-1], periods=10, freq='D'), exp_forecast, label='Forecast')
plt.title('Exponential Smoothing Model - Forecast')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()


# In[9]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate error metrics for ARIMA
arima_mae = mean_absolute_error(df['Ex_rate'][-10:], forecast)
arima_rmse = mean_squared_error(df['Ex_rate'][-10:], forecast, squared=False)

# Calculate error metrics for Exponential Smoothing
exp_mae = mean_absolute_error(df['Ex_rate'][-10:], exp_forecast)
exp_rmse = mean_squared_error(df['Ex_rate'][-10:], exp_forecast, squared=False)




# In[10]:


print(f"ARIMA MAE: {arima_mae}, RMSE: {arima_rmse}")
print(f"Exponential Smoothing MAE: {exp_mae}, RMSE: {exp_rmse}")


# Both ARIMA and Exponential Smoothing models offered valuable forecasts, with each model demonstrating specific advantages in different contexts. The ARIMA model excelled in capturing complex temporal dependencies, while the Exponential Smoothing model provided a more straightforward approach with effective parameter tuning. The choice between these models should be guided by the specific characteristics of the time series and the forecasting requirements.
# 
# This analysis underscores the importance of rigorous model selection and evaluation in time series forecasting. By leveraging both ARIMA and Exponential Smoothing techniques, we gain a more nuanced understanding of exchange rate dynamics and improve our ability to forecast future values with greater confidence.
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:




