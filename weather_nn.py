import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import glob

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

csv_files = glob.glob(os.path.join("data/Weather 2017", '*.csv'))
csv_files2 = glob.glob(os.path.join("data/Weather 2018", '*.csv'))
df1 = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
df2 = pd.concat([pd.read_csv(file) for file in csv_files2], ignore_index=True)
df = pd.concat([df1, df2], ignore_index=True)

# DATA CLEANUP AND PREPARATION
drop_cols = [
    "Longitude (x)",
    "Latitude (y)",
    "Station Name",
    "Climate ID",
    "Data Quality",
    "Max Temp Flag",
    "Min Temp Flag",
    "Mean Temp Flag",
    "Heat Deg Days Flag",
    "Cool Deg Days Flag",
    "Total Rain Flag",
    "Total Snow Flag",
    "Total Precip Flag",
    "Snow on Grnd Flag",
    "Dir of Max Gust Flag",
    "Spd of Max Gust Flag",
]
df = df.drop(columns=drop_cols)
# Replace "T" (trace amounts) and missing values
df.replace("T", 0.0, inplace=True)
df = df.fillna(0)

for col in df.columns:
    if col != "Date/Time":
        df[col] = pd.to_numeric(df[col])

print(df.columns)
# target mean temp is the result we want to predict
df["Target_Mean_Temp"] = df["Mean Temp (°C)"].shift(-1)
df["Date/Time"] = pd.to_datetime(df["Date/Time"])
df["Month"] = df["Date/Time"].dt.month
df["DayOfWeek"] = df["Date/Time"].dt.dayofweek

df = df.dropna()
x = df[
    [
        "Max Temp (°C)",
        "Min Temp (°C)",
        "Heat Deg Days (°C)",
        "Cool Deg Days (°C)",
        "Total Rain (mm)",
        "Total Snow (cm)",
        "Total Precip (mm)",
        "Snow on Grnd (cm)",
        "Spd of Max Gust (km/h)",
        "Month",
        "DayOfWeek",
        "Dir of Max Gust (10s deg)"
    ]
]
y = df["Target_Mean_Temp"]
# DATA SPLITTING
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, shuffle=False
)

# Normalize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# MODEL
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    BatchNormalization(),           # Normalize activations to speed up training and stabilize learning
    Dropout(0.3),                  # Randomly drop 30% of neurons during training to reduce overfitting

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),

    Dense(1)  # Output layer for regression
])

# Compile
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
# Train
model.fit(x_train, y_train, epochs=100, validation_split=0.2)

loss, mae = model.evaluate(x_test, y_test)
print(f"Mean Absolute Error on Test Set: {mae:.2f}°C")

# Predict tomorrow’s temp
preds = model.predict(x_test)
print(y_test)
for i in range(5):
    print(f"Predicted: {preds[i][0]:.2f}°C, Actual: {y_test.iloc[i]:.2f}°C")