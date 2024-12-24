import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import sys

# Generate random data for testing
"""
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=500, freq="D")
close = 10 + 10 * np.sin(2 * np.pi * dates.dayofyear / 500) + np.random.normal(0, 2, 500)
"""

# Read real data
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python test.py <ticker.csv>")


    data = pd.read_csv(sys.argv[1])
    dates = pd.to_datetime(data["Price"][2:].values)   
    close = data["Close"][2:].astype(float).values 
    close = np.round(close, 4)
    data = pd.DataFrame({"Date": dates, "Close": close})
    print(data.head())

    # Data visualization 
    plt.figure(figsize=(20, 30))
    sns.lineplot(data=data, x="Date", y="Close", color="blue")
    plt.title("Synthetic Close Time Series", fontsize=16)
    plt.xlabel("Date", fontsize=2)
    plt.ylabel("Close", fontsize=2)
    plt.grid()
    plt.show()

    # Prepare data for LSTM model
    class StockDataset(Dataset):
        def __init__(self, data, sequence_length=30):
            self.data = data
            self.sequence_length = sequence_length

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.sequence_length]
            y = self.data[idx + self.sequence_length]
            return torch.tensor(np.array(x, dtype=np.float32), dtype=torch.float32), torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float32)

    sequence_length = 30
    data_values = data["Close"].values
    dataset = StockDataset(data_values, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Construct LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=100, num_layers=3):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

    lstm_model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    # Train LSTM model and visualize training loss
    epochs = 300
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        lstm_model.train()
        for x, y in dataloader:
            x = x.unsqueeze(-1)
            y = y.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = lstm_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(dataloader))
        print(f"LSTM training progress: {round(((epoch + 1)/epochs*100), 1)} %  Loss: {round(losses[-1], 2)}")


    # Show training loss curve
    plt.figure(figsize=(20, 30))
    plt.plot(range(1, epochs + 1), losses, marker="o", color="green")
    plt.title("LSTM Training Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.grid()
    plt.show()

    # LSTM model feature extraction
    lstm_model.eval()
    lstm_features = []
    for i in range(len(data_values) - sequence_length):
        x = torch.tensor(data_values[i:i + sequence_length], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            lstm_features.append(lstm_model(x).item())

    # Random Forest model feature extraction and training
    lstm_features = np.array(lstm_features).reshape(-1, 1)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_size = int(len(lstm_features) * 0.8)
    rf_model.fit(lstm_features[:train_size], data_values[sequence_length:sequence_length + train_size])
 
    # Ramdom Forest model predictions on training and testing data
    rf_predictions_train = rf_model.predict(lstm_features[:train_size])
    actual_train = data_values[sequence_length:sequence_length + train_size]

    # Show Random Forest training predictions
    plt.figure(figsize=(20, 30))
    plt.plot(range(len(actual_train)), actual_train, label="Actual (Train)", color="blue")
    plt.plot(range(len(rf_predictions_train)), rf_predictions_train, label="RF Predicted (Train)", color="orange")
    plt.title("Random Forest Training Predictions", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Close", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # Show mixed predictions (LSTM + RF) on testing data
    rf_predictions = rf_model.predict(lstm_features[train_size:])
    actual = data_values[sequence_length + train_size:]

    plt.figure(figsize=(20, 30))
    plt.plot(range(len(actual)), actual, label="Actual (Test)", color="blue")
    plt.plot(range(len(rf_predictions)), rf_predictions, label="Predicted (LSTM + RF)", color="red")
    plt.title("LSTM + Random Forest Predictions", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Close", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

    # Use LSTM model to predict future 30 days
    future_features = []
    current_data = data_values[-(sequence_length):]  # Last 180 days of data

    for _ in range(30):  # predict 30 days
        x = torch.tensor(current_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            feature = lstm_model(x).item()
            future_features.append(feature)
        current_data = np.append(current_data[1:], feature)

    future_features = np.array(future_features).reshape(-1, 1)

    # Show mixed predictions (LSTM + RF) on testing data with future 30 days
    plt.figure(figsize=(20, 30))
    plt.plot(range(len(actual)), actual, label="Actual (Test)", color="blue")
    plt.plot(range(len(rf_predictions)), rf_predictions, label="Predicted (LSTM + RF)", color="red")
    plt.plot(range(len(actual), len(actual) + len(future_features)), future_features, label="Future Predictions (30 Days)", color="green")
    plt.title("LSTM + Random Forest Predictions with Future 30 Days", fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Close", fontsize=12)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()