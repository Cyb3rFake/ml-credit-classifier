"""Training script for credit classification model."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(filepath):
    """Load and prepare credit data."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    
    # Separate features and target
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk']
    
    return X, y

def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    logger.info("Training XGBoost classifier...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    return y_pred

if __name__ == '__main__':
    # Load data
    X, y = load_and_prepare_data('data/credit_data.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model and scaler
    joblib.dump(model, 'models/credit_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    logger.info("Model and scaler saved successfully")
