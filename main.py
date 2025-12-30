"""ML Credit Classifier - XGBoost Model for Credit Risk Assessment"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditClassifier:
    """
    Credit Risk Classification Model
    
    This class handles:
    - Data loading and preprocessing
    - Feature engineering
    - Model training
    - Model evaluation
    - Predictions
    """
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize the classifier
        
        Args:
            model_type (str): 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load credit data from CSV
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Data shape: {df.shape}")
        return df
    
    def preprocess(self, df, target_col='default'):
        """
        Preprocess the data
        
        Args:
            df (pd.DataFrame): Raw data
            target_col (str): Target column name
            
        Returns:
            tuple: (X, y) - features and target
        """
        logger.info("Starting preprocessing...")
        
        # Handle missing values
        df = df.dropna()
        logger.info(f"Shape after dropping NaN: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        logger.info(f"Final features: {X.shape[1]}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
        """
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        logger.info("Model training completed!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Performance metrics
        """
        logger.info("Evaluating model...")
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            'auc_score': auc_score,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"AUC Score: {auc_score:.4f}")
        logger.info(f"\n{metrics['classification_report']}")
        
        return metrics
    
    def predict(self, X_new):
        """
        Make predictions on new data
        
        Args:
            X_new (pd.DataFrame): New features
            
        Returns:
            np.array: Predictions
        """
        X_scaled = self.scaler.transform(X_new)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        joblib.dump(self.model, filepath)
        joblib.dump(self.scaler, filepath.replace('.pkl', '_scaler.pkl'))
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to model file
        """
        self.model = joblib.load(filepath)
        self.scaler = joblib.load(filepath.replace('.pkl', '_scaler.pkl'))
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    classifier = CreditClassifier(model_type='xgboost')
    
    # Load and preprocess data
    df = classifier.load_data('data/credit_data.csv')
    X, y = classifier.preprocess(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    
    # Save model
    classifier.save_model('models/credit_classifier.pkl')
