import pandas as pd
import joblib  # For loading the pre-trained model
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'dataset/timeseries_xlsx_table.xlsx'  # Update this path
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(data.head())

# Preprocessing
target_column = 'accel_x'  # Update this to your actual target column name if applicable
X = data.drop(columns=[target_column])  # Features (all columns except target)

# Load the pre-trained model (ensure you have a saved model file)
model_path = 'path_to_your_model/pretrained_model.pkl'  # Update this path
model = joblib.load(model_path)

# Make predictions on the entire dataset or a specific test set
y_pred = model.predict(X)

# Evaluate the model performance on the existing data (if applicable)
if target_column in data.columns:
    y_true = data[target_column]
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

# Display coefficients if applicable (for linear models)
if hasattr(model, 'coef_'):
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print("Model Coefficients:")
    print(coefficients)

# Making predictions for new data (example)
new_data = pd.DataFrame({
    'accel_x': [1.0],  # Replace with actual values for prediction
    'accel_y': [0.5],  # Replace with actual values for prediction
    'accel_z': [0.3],  # Replace with actual values for prediction
    'ambient_light': [200],  # Replace with actual values for prediction
    'blue_proportion': [50],  # Replace with actual values for prediction
    'green_proportion': [30],  # Replace with actual values for prediction
    'gyro_x': [0.1],  # Replace with actual values for prediction
    'gyro_y': [0.2],  # Replace with actual values for prediction
    'gyro_z': [0.3],  # Replace with actual values for prediction
    'pressure_kPa': [101.3],  # Replace with actual values for prediction
    'pressure_mmHg': [760],  # Replace with actual values for prediction
    'proximity': [10],  # Replace with actual values for prediction
    'red_proportion': [20],  # Replace with actual values for prediction
    'temperature': [22],  # Replace with actual values for prediction
    'temperature_C': [22],  # Replace with actual values for prediction
    'temperature_F': [71.6],  # Replace with actual values for prediction
    'tilt_x': [0.0],  # Replace with actual values for prediction
    'tilt_y': [0.0],  # Replace with actual values for prediction
    'tilt_z': [0.0]   # Replace with actual values for prediction
})

predicted_value = model.predict(new_data)
print(f'Predicted Value for New Data: {predicted_value[0]}')
