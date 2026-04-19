"""
S&P 500 Stock Price Prediction (Regression with Price Features)
===============================================================
Task: Predict actual closing price

Models included:
- Traditional ML: Random Forest, XGBoost
- Deep Learning: LSTM, Transformer
- Ensemble: Voting Regressor

NOTE: This version INCLUDES price features (MA, EMA, BB, Close_lag, etc.)

Author: Student
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# For advanced models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using alternative models.")

# Deep Learning Libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

SELECTED_STOCKS = ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'JPM']

DATA_PATH = 'c:/Users/TK/Desktop/5564/project/all_stocks_5yr.csv'
INDIVIDUAL_STOCK_PATH = 'c:/Users/TK/Desktop/5564/project/individual_stocks_5yr/'

# Deep Learning Config
DL_CONFIG = {
    'sequence_length': 20,
    'hidden_size': 64,
    'num_layers': 2,
    'd_model': 64,
    'nhead': 4,
    'num_encoder_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.2,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001
}

# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

class LSTMRegressor(nn.Module):
    """LSTM模型用于回归"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    """Transformer模型用于回归"""
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# =============================================================================
# PART 1: DATA LOADING
# =============================================================================

def load_data():
    """Load the S&P 500 stock data"""
    print("=" * 60)
    print("PART 1: DATA LOADING AND EXPLORATION")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of Unique Stocks: {df['Name'].nunique()}")
    
    return df

def get_stock_data(df, stock_name):
    """Extract data for a specific stock"""
    stock_df = df[df['Name'] == stock_name].copy()
    stock_df = stock_df.sort_values('date').reset_index(drop=True)
    return stock_df

# =============================================================================
# PART 2: FEATURE ENGINEERING
# =============================================================================

def add_technical_indicators(df):
    """Add technical indicators as features"""
    df = df.copy()
    
    # Moving Averages
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Price Rate of Change
    df['ROC5'] = df['close'].pct_change(periods=5) * 100
    df['ROC10'] = df['close'].pct_change(periods=10) * 100
    df['ROC20'] = df['close'].pct_change(periods=20) * 100
    
    # Volatility
    df['STD5'] = df['close'].rolling(window=5).std()
    df['STD10'] = df['close'].rolling(window=10).std()
    df['STD20'] = df['close'].rolling(window=20).std()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # Volume indicators
    df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
    df['Volume_MA10'] = df['volume'].rolling(window=10).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_MA5']
    
    # Price differences
    df['High_Low_diff'] = df['high'] - df['low']
    df['Open_Close_diff'] = df['close'] - df['open']
    df['High_Close_diff'] = df['high'] - df['close']
    df['Low_Close_diff'] = df['close'] - df['low']
    
    # Daily returns
    df['Daily_Return'] = df['close'].pct_change() * 100
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'Close_lag{lag}'] = df['close'].shift(lag)
        df[f'Return_lag{lag}'] = df['Daily_Return'].shift(lag)
        df[f'Volume_lag{lag}'] = df['volume'].shift(lag)
    
    return df

def prepare_features_targets(df):
    """Prepare features and targets for regression (INCLUDE price features)"""
    df = df.copy()
    
    df['Next_Close'] = df['close'].shift(-1)
    df = df.dropna()
    
    # Only exclude non-feature columns (INCLUDE price-related features)
    exclude_cols = ['date', 'Name', 'Next_Close']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['Next_Close'].values
    
    return X, y, feature_cols, df

# =============================================================================
# PART 3: TRADITIONAL ML MODELS
# =============================================================================

def train_traditional_models(X_train, X_test, y_train, y_test):
    """Train traditional ML models (Random Forest, XGBoost only)"""
    print("\n" + "=" * 60)
    print("TRAINING TRADITIONAL ML MODELS")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'predictions': y_pred,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print(f"  RMSE: {results[name]['rmse']:.4f}, R²: {results[name]['r2']:.4f}")
    
    return results, scaler

# =============================================================================
# PART 4: DEEP LEARNING MODELS
# =============================================================================

def create_sequences(X, y, seq_length):
    """Create sequences for deep learning"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def train_dl_model(model, train_loader, val_loader, epochs, device):
    """Train deep learning model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_CONFIG['learning_rate'])
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    return model

def train_deep_learning_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train LSTM and Transformer models"""
    print("\n" + "=" * 60)
    print("TRAINING DEEP LEARNING MODELS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Normalize
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Create sequences
    seq_length = DL_CONFIG['sequence_length']
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)
    
    # Adjust y for sequences
    y_test_adj = y_test[seq_length:]
    
    # Validation split
    val_size = int(len(X_train_seq) * 0.2)
    X_train_dl, X_val = X_train_seq[val_size:], X_train_seq[:val_size]
    y_train_dl, y_val = y_train_seq[val_size:], y_train_seq[:val_size]
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_dl)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test_seq)
    y_train_t = torch.FloatTensor(y_train_dl)
    y_val_t = torch.FloatTensor(y_val)
    
    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=DL_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=DL_CONFIG['batch_size'])
    test_loader = DataLoader(TensorDataset(X_test_t, torch.FloatTensor(y_test_seq)), 
                             batch_size=DL_CONFIG['batch_size'])
    
    input_size = len(feature_cols)
    results = {}
    
    # LSTM
    print("\nTraining LSTM...")
    lstm_model = LSTMRegressor(input_size, DL_CONFIG['hidden_size'], DL_CONFIG['num_layers'], 
                               DL_CONFIG['dropout']).to(device)
    lstm_model = train_dl_model(lstm_model, train_loader, val_loader, DL_CONFIG['epochs'], device)
    
    lstm_model.eval()
    lstm_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            lstm_preds.extend(outputs.cpu().numpy())
    lstm_preds = np.array(lstm_preds)
    lstm_preds_orig = scaler_y.inverse_transform(lstm_preds)
    
    results['LSTM'] = {
        'predictions': lstm_preds_orig.flatten(),
        'rmse': np.sqrt(mean_squared_error(y_test_adj, lstm_preds_orig)),
        'mae': mean_absolute_error(y_test_adj, lstm_preds_orig),
        'r2': r2_score(y_test_adj, lstm_preds_orig),
        'mape': np.mean(np.abs((y_test_adj - lstm_preds_orig.flatten()) / y_test_adj)) * 100
    }
    print(f"  RMSE: {results['LSTM']['rmse']:.4f}, R²: {results['LSTM']['r2']:.4f}")
    
    # Transformer
    print("\nTraining Transformer...")
    transformer_model = TransformerRegressor(input_size, DL_CONFIG['d_model'], DL_CONFIG['nhead'],
                                             DL_CONFIG['num_encoder_layers'], DL_CONFIG['dim_feedforward'],
                                             DL_CONFIG['dropout']).to(device)
    transformer_model = train_dl_model(transformer_model, train_loader, val_loader, DL_CONFIG['epochs'], device)
    
    transformer_model.eval()
    trans_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = transformer_model(X_batch)
            trans_preds.extend(outputs.cpu().numpy())
    trans_preds = np.array(trans_preds)
    trans_preds_orig = scaler_y.inverse_transform(trans_preds)
    
    results['Transformer'] = {
        'predictions': trans_preds_orig.flatten(),
        'rmse': np.sqrt(mean_squared_error(y_test_adj, trans_preds_orig)),
        'mae': mean_absolute_error(y_test_adj, trans_preds_orig),
        'r2': r2_score(y_test_adj, trans_preds_orig),
        'mape': np.mean(np.abs((y_test_adj - trans_preds_orig.flatten()) / y_test_adj)) * 100
    }
    print(f"  RMSE: {results['Transformer']['rmse']:.4f}, R²: {results['Transformer']['r2']:.4f}")
    
    return results

# =============================================================================
# PART 5: ENSEMBLE METHODS
# =============================================================================

def ensemble_regression(X_train, X_test, y_train, y_test, scaler):
    """Ensemble regression with voting"""
    print("\n" + "=" * 60)
    print("ENSEMBLE REGRESSION")
    print("=" * 60)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    if XGBOOST_AVAILABLE:
        estimators.append(('xgb', XGBRegressor(n_estimators=100, max_depth=5, random_state=42)))
    
    voting_reg = VotingRegressor(estimators=estimators)
    voting_reg.fit(X_train_scaled, y_train)
    y_pred = voting_reg.predict(X_test_scaled)
    
    results = {
        'predictions': y_pred,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    print(f"Voting Ensemble: RMSE={results['rmse']:.4f}, R²={results['r2']:.4f}")
    
    return results

# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

def plot_results(y_test, results, stock_name):
    """Plot regression results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Regression Results (with Price Features) - {stock_name}', fontsize=14, fontweight='bold')
    
    # Model Comparison
    ax1 = axes[0]
    models = list(results.keys())
    rmse_values = [results[m]['rmse'] for m in models]
    r2_values = [results[m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, rmse_values, width, label='RMSE', color='steelblue')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('RMSE', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    ax1b = ax1.twinx()
    ax1b.bar(x + width/2, r2_values, width, label='R²', color='coral')
    ax1b.set_ylabel('R²', color='coral')
    ax1b.tick_params(axis='y', labelcolor='coral')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_title('Model Performance Comparison')
    
    # Actual vs Predicted for best model
    ax2 = axes[1]
    best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
    y_pred = results[best_model]['predictions']
    
    # Adjust lengths
    min_len = min(len(y_test), len(y_pred))
    y_test_plot = y_test[:min_len]
    y_pred_plot = y_pred[:min_len]
    
    ax2.scatter(y_test_plot, y_pred_plot, alpha=0.5, s=10)
    ax2.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'Actual vs Predicted - {best_model}')
    
    plt.tight_layout()
    plt.savefig(f'c:/Users/TK/Desktop/5564/project/results/regression_with_price_{stock_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved for {stock_name}")

# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("S&P 500 STOCK PRICE PREDICTION (REGRESSION WITH PRICE FEATURES)")
    print("=" * 70)
    print("\nModels: Random Forest, XGBoost, LSTM, Transformer, Voting")
    print("Features: INCLUDING price-based features (MA, EMA, BB, Close_lag)")
    print("=" * 70)
    
    import os
    os.makedirs('c:/Users/TK/Desktop/5564/project/results', exist_ok=True)
    
    # Initialize results list for CSV export
    csv_results = []
    
    df = load_data()
    available_stocks = [s for s in SELECTED_STOCKS if s in df['Name'].unique()]
    print(f"\nAvailable stocks: {available_stocks}")
    
    all_results = {}
    
    for stock in available_stocks[:5]:
        print("\n" + "=" * 70)
        print(f"PROCESSING: {stock}")
        print("=" * 70)
        
        stock_df = get_stock_data(df, stock)
        if len(stock_df) < 100:
            continue
        
        stock_df = add_technical_indicators(stock_df)
        X, y, feature_cols, _ = prepare_features_targets(stock_df)
        
        print(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        print(f"Feature columns: {feature_cols[:10]}...")  # Show first 10 features
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train traditional ML models
        ml_results, scaler = train_traditional_models(X_train, X_test, y_train, y_test)
        
        # Train deep learning models
        dl_results = train_deep_learning_models(X_train, X_test, y_train, y_test, feature_cols)
        
        # Ensemble
        ensemble_results = ensemble_regression(X_train, X_test, y_train, y_test, scaler)
        ml_results['Voting Ensemble'] = ensemble_results
        
        # Combine results
        results = {**ml_results, **dl_results}
        
        # Plot
        plot_results(y_test, results, stock)
        
        all_results[stock] = results
        
        # Collect results for CSV
        for model_name, metrics in results.items():
            csv_results.append({
                'Stock': stock,
                'Model': model_name,
                'RMSE': metrics.get('rmse', np.nan),
                'MAE': metrics.get('mae', np.nan),
                'R2': metrics.get('r2', np.nan),
                'MAPE': metrics.get('mape', np.nan)
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(csv_results)
    results_df.to_csv('c:/Users/TK/Desktop/5564/project/results/regression_results.csv', index=False)
    print(f"\nResults saved to c:/Users/TK/Desktop/5564/project/results/regression_results.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    print(f"\n{'Stock':<10} {'Best Model':<20} {'RMSE':<12} {'R²':<12}")
    print("-" * 54)
    for stock, results in all_results.items():
        best = min(results.keys(), key=lambda x: results[x].get('rmse', float('inf')))
        print(f"{stock:<10} {best:<20} {results[best]['rmse']:<12.4f} {results[best]['r2']:<12.4f}")
    
    print("\n" + "=" * 70)
    print("REGRESSION TASK (WITH PRICE FEATURES) COMPLETED!")
    print("=" * 70)
    
    return all_results

if __name__ == "__main__":
    results = main()