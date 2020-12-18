import pandas as pd
import numpy as np

series = pd.read_csv('data/passengers.csv',sep=';', header=0, index_col=0, squeeze=True)
data_ARIMA_cut = series.iloc[0:125,]
data_ARIMA_cut_float = data_ARIMA_cut[:].astype(np.float)
data_ARIMA_cut_float.tail()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data_ARIMA_cut_float)
plot_pacf(data_ARIMA_cut_float)


diff_1=data_ARIMA_cut_float.diff(periods=1).iloc[1:]
plot_acf(diff_1)
plot_pacf(diff_1)

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(data_ARIMA_cut_float, order=(1,1,0))
model_fit = model.fit(trend='nc',full_output=True, disp=1)
print(model_fit.summary())

fore = model_fit.forecast(steps=1)
print(fore)s