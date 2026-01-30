import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """Train and evaluate ML models for NLP tasks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.best_params = None
        logger.info("ModelTrainer initialized")
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def get_model(self, model_type: str):
        """Get model instance by type."""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boost': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42)
        }
        return models.get(model_type.lower())
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'logistic'):
        """Train model on training data."""
        logger.info(f"Training {model_type} model")
        self.model = self.get_model(model_type)
        if self.model is None:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train, y_train)
        logger.info(f"{model_type} model training complete")
        return self.model
    
    def train_with_grid_search(self, X_train: np.ndarray, y_train: np.ndarray, 
                               model_type: str = 'logistic', param_grid: Dict = None):
        """Train model with hyperparameter tuning."""
        logger.info(f"Training {model_type} with grid search")
        base_model = self.get_model(model_type)
        
        if param_grid is None:
            # Default param grids
            param_grids = {
                'logistic': {'C': [0.1, 1, 10], 'penalty': ['l2']},
                'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
                'gradient_boost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            }
            param_grid = param_grids.get(model_type.lower(), {})
        
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        logger.info(f"Best parameters: {self.best_params}")
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Evaluating model")
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

if __name__ == "__main__":
    # Example usage
    config = {}
    trainer = ModelTrainer(config)
    print("Model trainer ready")
