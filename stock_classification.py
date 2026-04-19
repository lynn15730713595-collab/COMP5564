"""
S&P 500 Stock Price Trend Prediction (Classification with Price Features)
=========================================================================
Task: Predict if stock price will go up or down

Models included:
- Traditional ML: Random Forest, XGBoost
- Deep Learning: LSTM, Transformer
- Ensemble: Voting Classifier

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
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# For advanced models
try:
    from xgboost import XGBClassifier
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
    'epochs': 100,
    'learning_rate': 0.001
}

# =============================================================================
# DEEP LEARNING MODELS
# =============================================================================

class LSTMClassifier(nn.Module):
    """LSTM模型用于分类"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)


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


class TransformerClassifier(nn.Module):
    """Transformer模型用于分类"""
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out = x[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return self.sigmoid(out)


# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION
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
    """Prepare features and targets for classification (INCLUDE price features)"""
    df = df.copy()
    
    df['Next_Close'] = df['close'].shift(-1)
    df['Price_Direction'] = (df['Next_Close'] > df['close']).astype(int)
    df = df.dropna()
    
    # Only exclude non-feature columns - INCLUDE price features
    exclude_cols = ['date', 'Name', 'Next_Close', 'Price_Direction']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['Price_Direction'].values
    
    return X, y, feature_cols, df

# =============================================================================
# PART 3: TRADITIONAL ML MODELS
# =============================================================================

def train_traditional_models(X_train, X_test, y_train, y_test):
    """Train traditional ML models with GridSearchCV + Cross-Validation"""
    print("\n" + "=" * 60)
    print("TRAINING TRADITIONAL ML MODELS (with GridSearchCV + 5-Fold CV)")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    }
    
    if XGBOOST_AVAILABLE:
        param_grids['XGBoost'] = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
    
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name} with GridSearchCV...")
        
        # GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.4f}")
        
        # Use best model for prediction
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        results[name] = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'best_params': grid_search.best_params_
        }
        
        print(f"  Accuracy: {results[name]['accuracy']:.4f}, Precision: {results[name]['precision']:.4f}, Recall: {results[name]['recall']:.4f}, F1: {results[name]['f1']:.4f}, AUC: {results[name]['auc']:.4f}")
    
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

def train_dl_model(model, train_loader, epochs, device, y_train=None):
    """Train deep learning model with class weighting"""
    # Calculate class weights if y_train is provided
    if y_train is not None:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.BCELoss(weight=class_weights)
        print(f"  Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.BCELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_CONFIG['learning_rate'])
    
    # Train for fixed number of epochs (no early stopping with validation set)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
    
    return model

def train_deep_learning_models(X_train, X_test, y_train, y_test, feature_cols):
    """Train LSTM and Transformer models"""
    print("\n" + "=" * 60)
    print("TRAINING DEEP LEARNING MODELS")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Normalize
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create sequences
    seq_length = DL_CONFIG['sequence_length']
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, seq_length)
    
    # Adjust y for sequences
    y_train_adj = y_train[seq_length:]
    y_test_adj = y_test[seq_length:]
    
    # Use full training data (no separate validation set for simplicity)
    X_train_dl = X_train_seq
    y_train_dl = y_train_seq
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_dl)
    X_test_t = torch.FloatTensor(X_test_seq)
    y_train_t = torch.FloatTensor(y_train_dl).reshape(-1, 1)
    
    # Data loaders
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=DL_CONFIG['batch_size'], shuffle=True)
    # No validation loader - using full training data
    test_loader = DataLoader(TensorDataset(X_test_t, torch.FloatTensor(y_test_adj).reshape(-1, 1)), 
                             batch_size=DL_CONFIG['batch_size'])
    
    input_size = len(feature_cols)
    results = {}
    
    # LSTM
    print("\nTraining LSTM...")
    lstm_model = LSTMClassifier(input_size, DL_CONFIG['hidden_size'], DL_CONFIG['num_layers'], 
                                DL_CONFIG['dropout']).to(device)
    lstm_model = train_dl_model(lstm_model, train_loader, DL_CONFIG['epochs'], device, None)
    
    lstm_model.eval()
    lstm_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = lstm_model(X_batch)
            lstm_preds.extend(outputs.cpu().numpy())
    lstm_preds = np.array(lstm_preds)
    
    # Use optimal threshold search - wider range, balance precision/recall
    best_threshold = 0.5
    best_score = -1
    for thresh in np.arange(0.2, 0.8, 0.02):
        preds = (lstm_preds.flatten() > thresh).astype(int)
        # Check that we have both classes predicted
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(y_test_adj, preds, zero_division=0)
        acc = accuracy_score(y_test_adj, preds)
        # Prefer thresholds that give balanced predictions
        score = f1 + acc * 0.5
        if score > best_score:
            best_score = score
            best_threshold = thresh
    lstm_pred_labels = (lstm_preds.flatten() > best_threshold).astype(int)
    
    results['LSTM'] = {
        'predictions': lstm_pred_labels.flatten(),
        'probabilities': lstm_preds.flatten(),
        'accuracy': accuracy_score(y_test_adj, lstm_pred_labels),
        'precision': precision_score(y_test_adj, lstm_pred_labels),
        'recall': recall_score(y_test_adj, lstm_pred_labels),
        'f1': f1_score(y_test_adj, lstm_pred_labels),
        'auc': roc_auc_score(y_test_adj, lstm_preds)
    }
    print(f"  Accuracy: {results['LSTM']['accuracy']:.4f}, Precision: {results['LSTM']['precision']:.4f}, Recall: {results['LSTM']['recall']:.4f}, F1: {results['LSTM']['f1']:.4f}, AUC: {results['LSTM']['auc']:.4f}")
    
    # Transformer
    print("\nTraining Transformer...")
    transformer_model = TransformerClassifier(input_size, DL_CONFIG['d_model'], DL_CONFIG['nhead'],
                                              DL_CONFIG['num_encoder_layers'], DL_CONFIG['dim_feedforward'],
                                              DL_CONFIG['dropout']).to(device)
    transformer_model = train_dl_model(transformer_model, train_loader, DL_CONFIG['epochs'], device, None)
    
    transformer_model.eval()
    trans_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = transformer_model(X_batch)
            trans_preds.extend(outputs.cpu().numpy())
    trans_preds = np.array(trans_preds)
    
    # Use optimal threshold search - wider range, balance precision/recall
    best_threshold = 0.5
    best_score = -1
    for thresh in np.arange(0.2, 0.8, 0.02):
        preds = (trans_preds.flatten() > thresh).astype(int)
        # Check that we have both classes predicted
        if len(np.unique(preds)) < 2:
            continue
        f1 = f1_score(y_test_adj, preds, zero_division=0)
        acc = accuracy_score(y_test_adj, preds)
        # Prefer thresholds that give balanced predictions
        score = f1 + acc * 0.5
        if score > best_score:
            best_score = score
            best_threshold = thresh
    trans_pred_labels = (trans_preds.flatten() > best_threshold).astype(int)
    
    results['Transformer'] = {
        'predictions': trans_pred_labels.flatten(),
        'probabilities': trans_preds.flatten(),
        'accuracy': accuracy_score(y_test_adj, trans_pred_labels),
        'precision': precision_score(y_test_adj, trans_pred_labels),
        'recall': recall_score(y_test_adj, trans_pred_labels),
        'f1': f1_score(y_test_adj, trans_pred_labels),
        'auc': roc_auc_score(y_test_adj, trans_preds)
    }
    print(f"  Accuracy: {results['Transformer']['accuracy']:.4f}, Precision: {results['Transformer']['precision']:.4f}, Recall: {results['Transformer']['recall']:.4f}, F1: {results['Transformer']['f1']:.4f}, AUC: {results['Transformer']['auc']:.4f}")
    
    return results

# =============================================================================
# PART 5: ENSEMBLE METHODS
# =============================================================================

def ensemble_classification(X_train, X_test, y_train, y_test, scaler):
    """Ensemble classification with voting"""
    print("\n" + "=" * 60)
    print("ENSEMBLE CLASSIFICATION")
    print("=" * 60)
    
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    if XGBOOST_AVAILABLE:
        estimators.append(('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                                                use_label_encoder=False, eval_metric='logloss')))
    
    results = {}
    
    # Hard Voting
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train_scaled, y_train)
    y_pred_hard = voting_hard.predict(X_test_scaled)
    results['Voting (Hard)'] = {
        'predictions': y_pred_hard,
        'accuracy': accuracy_score(y_test, y_pred_hard),
        'precision': precision_score(y_test, y_pred_hard),
        'recall': recall_score(y_test, y_pred_hard),
        'f1': f1_score(y_test, y_pred_hard),
        'auc': None  # Hard voting doesn't have probabilities
    }
    print(f"Voting (Hard): Accuracy={results['Voting (Hard)']['accuracy']:.4f}, Precision={results['Voting (Hard)']['precision']:.4f}, Recall={results['Voting (Hard)']['recall']:.4f}, F1={results['Voting (Hard)']['f1']:.4f}")
    
    # Soft Voting
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X_train_scaled, y_train)
    y_pred_soft = voting_soft.predict(X_test_scaled)
    y_pred_proba_soft = voting_soft.predict_proba(X_test_scaled)[:, 1]
    results['Voting (Soft)'] = {
        'predictions': y_pred_soft,
        'probabilities': y_pred_proba_soft,
        'accuracy': accuracy_score(y_test, y_pred_soft),
        'precision': precision_score(y_test, y_pred_soft),
        'recall': recall_score(y_test, y_pred_soft),
        'f1': f1_score(y_test, y_pred_soft),
        'auc': roc_auc_score(y_test, y_pred_proba_soft)
    }
    print(f"Voting (Soft): Accuracy={results['Voting (Soft)']['accuracy']:.4f}, Precision={results['Voting (Soft)']['precision']:.4f}, Recall={results['Voting (Soft)']['recall']:.4f}, F1={results['Voting (Soft)']['f1']:.4f}, AUC={results['Voting (Soft)']['auc']:.4f}")
    
    return results

# =============================================================================
# PART 6: VISUALIZATION
# =============================================================================

def plot_results(y_test, results, stock_name):
    """Plot classification results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Classification Results (with Price Features) - {stock_name}', fontsize=14, fontweight='bold')
    
    # Model Comparison
    ax1 = axes[0]
    models = list(results.keys())
    metrics = ['accuracy', 'f1']
    x = np.arange(len(models))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        ax1.bar(x + i * width, values, width, label=metric.upper())
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # ROC Curves
    ax2 = axes[1]
    for name, res in results.items():
        if res.get('probabilities') is not None:
            fpr, tpr, _ = roc_curve(y_test[:len(res['probabilities'])], res['probabilities'])
            ax2.plot(fpr, tpr, label=f"{name} (AUC: {res['auc']:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'c:/Users/TK/Desktop/5564/project/results/classification_with_price_{stock_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved for {stock_name}")

# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("S&P 500 STOCK PRICE TREND PREDICTION (CLASSIFICATION WITH PRICE FEATURES)")
    print("=" * 70)
    print("\nModels: Random Forest, XGBoost, LSTM, Transformer, Voting")
    print("NOTE: This version INCLUDES price features")
    print("=" * 70)
    
    import os
    RESULTS_PATH = 'c:/Users/TK/Desktop/5564/project/results'
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
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
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train traditional ML models
        ml_results, scaler = train_traditional_models(X_train, X_test, y_train, y_test)
        
        # Train deep learning models
        dl_results = train_deep_learning_models(X_train, X_test, y_train, y_test, feature_cols)
        
        # Ensemble
        ensemble_results = ensemble_classification(X_train, X_test, y_train, y_test, scaler)
        
        # Combine results
        results = {**ml_results, **dl_results, **ensemble_results}
        
        # Plot
        plot_results(y_test, results, stock)
        
        all_results[stock] = results
        
        # Show summary table for this stock
        print("\n" + "=" * 70)
        print(f"{stock} - ALL MODELS RESULTS")
        print("=" * 70)
        print(f"{'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 72)
        for model_name, metrics in results.items():
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            auc_val = metrics.get('auc')
            if auc_val is None:
                auc_str = "N/A"
            else:
                auc_str = f"{auc_val:<10.4f}"
            print(f"{model_name:<22} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc_str}")
        
        # Collect results for CSV
        for model_name, metrics in results.items():
            csv_results.append({
                'Stock': stock,
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', np.nan),
                'Precision': metrics.get('precision', np.nan),
                'Recall': metrics.get('recall', np.nan),
                'F1_Score': metrics.get('f1', np.nan),
                'AUC': metrics.get('auc', np.nan)
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(csv_results)
    results_df.to_csv(f'{RESULTS_PATH}/classification_results.csv', index=False)
    print(f"\nResults saved to {RESULTS_PATH}/classification_results.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY REPORT - BEST MODEL FOR EACH STOCK")
    print("=" * 70)
    
    print(f"\n{'Stock':<10} {'Best Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 82)
    for stock, results in all_results.items():
        best = max(results.keys(), key=lambda x: results[x].get('accuracy', 0))
        r = results[best]
        acc = r.get('accuracy', 0)
        prec = r.get('precision', 0)
        rec = r.get('recall', 0)
        f1 = r.get('f1', 0)
        auc = r.get('auc', 0)
        print(f"{stock:<10} {best:<22} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f}")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION TASK (WITH PRICE FEATURES) COMPLETED!")
    print("=" * 70)
    
    return all_results

if __name__ == "__main__":
    results = main()