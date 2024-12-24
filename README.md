# LSTM-ARIMA for Time Series Data Prediction: A Toy Model Analysis
A toy model analysis that demonstrates the integration of deep learning and machine learning techniques.

## 1. Objective
The primary aim of this project is to build a toy model that demonstrates the integration of deep learning and machine learning techniques for stock market prediction. The model uses historical stock price data to predict future trends, focusing on the development and evaluation of a proof-of-concept pipeline rather than providing actionable financial insights.

## 2. Methodology

### 2.1 Data Acquisition
The script `get_data.py` facilitates the acquisition of stock price data using the Yahoo Finance API (`yfinance`). Data is downloaded by specifying a stock ticker, with the default range starting from January 1, 2020, to the present date. The data is saved as a CSV file for subsequent processing.

### 2.2 Data Processing
In the `test.py` script:
- Historical stock prices are read from a CSV file.
- A subset of data is visualized to understand trends in the time series.

### 2.3 Hybrid Model Design
The model combines:
1. **LSTM (Long Short-Term Memory)**: Used for feature extraction from the time series data.
   - **Architecture**: A 3-layer LSTM with 100 hidden units followed by a fully connected layer.
   - **Training**: Optimized using Adam, with mean squared error (MSE) as the loss function.
   - **Loss Convergence**: Loss decreases over 300 epochs, confirming convergence during training.
2. **Random Forest Regressor**: Predicts future prices using features extracted by the LSTM model.
   - Training data consists of 80% of the extracted features, while the rest is used for testing.

### 2.4 Model Evaluation
- Training loss curve visualization confirms effective learning during the LSTM training phase.
- Predictions on test data show the combined (LSTM + Random Forest) model's capability to capture general trends.
- The model is extended to forecast stock prices for the next 30 days.

### 2.5 Visualization
Multiple visualizations are generated:
- Time series trends of historical prices.
- Loss curves during the LSTM training process.
- Random Forest predictions compared to actual values on training and test datasets.
- Future stock price predictions using the hybrid model.

## 3. Results
- The LSTM model successfully extracts meaningful features from the input data, as demonstrated by its convergence.
- The hybrid model provides reasonable approximations of stock price trends but lacks the robustness for real-world predictions due to the simplicity of the dataset and the model design.

## 4. Limitations
- **Simplistic Assumptions**: The toy model does not account for macroeconomic factors or external market influences.
- **Limited Scope**: Predictions are based solely on historical prices, without additional data sources such as trading volumes or news sentiment.
- **Overfitting Risks**: The model may overfit to the limited data used for training.

## 5. Disclaimer
This model is intended for educational purposes only. While the loss function converges and the hybrid approach demonstrates potential, the predictions are not reliable for actual stock market investment decisions.
