import pandas as pd
from fbprophet import Prophet

# Load the dataset
df = pd.read_excel('/dataset/timeseries_xlsx_table.xlsx')

# Prepare the data for Prophet
# Assuming 'timestamp' is the column with the time information
df.rename(columns={'timestamp': 'ds', 'temperature': 'y'}, inplace=True)

# Initialize the Prophet model
model = Prophet()

# Fit the model
model.fit(df[['ds', 'y']])

# Make future dataframe
future = model.make_future_dataframe(periods=365)  # Predicting for the next 365 days

# Predict future data
forecast = model.predict(future)

# Save the forecast to a new Excel file
forecast.to_excel('forecasted_data.xlsx', index=False)

# Print meaningful inferences
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecast
fig = model.plot(forecast)
fig.show()