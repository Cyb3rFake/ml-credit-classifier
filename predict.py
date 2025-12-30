"""Prediction script for credit classification model."""
import joblib
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CreditPredictor:
    """Predict credit risk using trained model."""
    
    def __init__(self, model_path, scaler_path):
        """Initialize predictor with model and scaler.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to feature scaler file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Scaler loaded from {scaler_path}")
    
    def predict(self, data):
        """Make predictions on input data.
        
        Args:
            data: Input features (DataFrame or array)
        
        Returns:
            Predictions and probabilities
        """
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def predict_single(self, features):
        """Predict for a single sample.
        
        Args:
            features: Dict or array of features
        
        Returns:
            Risk prediction and probability
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = pd.DataFrame([features])
        
        pred, prob = self.predict(df)
        return {
            'risk': 'High' if pred[0] == 1 else 'Low',
            'probability': prob[0][pred[0]],
            'raw_prediction': int(pred[0])
        }

def main():
    """Example usage of predictor."""
    # Initialize predictor
    predictor = CreditPredictor(
        'models/credit_classifier.pkl',
        'models/scaler.pkl'
    )
    
    # Example: Predict on new sample
    sample = {
        'age': 35,
        'income': 50000,
        'credit_score': 720,
        'debt_ratio': 0.3,
        'num_accounts': 4
    }
    
    result = predictor.predict_single(sample)
    logger.info(f"Prediction result: {result}")
    
    return result

if __name__ == '__main__':
    main()
