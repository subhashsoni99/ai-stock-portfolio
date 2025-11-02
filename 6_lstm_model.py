import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import accuracy_score

# Prepare Data for LSTM
# Load your prepared features (from previous day)
data = pd.read_csv("data/features_TCS.csv", index_col=0)
print(data.head())

# Select features and target
features = ['Close', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI_14']
target = 'Target'

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

X, y = [], []
window = 10  # lookback window (10 days)

for i in range(window, len(scaled_data)):
    X.append(scaled_data[i-window:i])
    y.append(data[target].iloc[i])

X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(X_train.shape, X_test.shape)

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the Model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# Evaluate
preds = (model.predict(X_test) > 0.5).astype(int)
acc = accuracy_score(y_test, preds)
print("LSTM Accuracy:", acc)

# Visualize
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('LSTM Model Accuracy')
plt.show()
