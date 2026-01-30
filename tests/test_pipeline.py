import unittest
import numpy as np
import pandas as pd
from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.preprocessing import TextPreprocessor
from src.pipeline.feature_extraction import FeatureExtractor
from src.pipeline.model_training import ModelTrainer

class TestNLPPipeline(unittest.TestCase):
    """Test cases for NLP pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {'data_dir': 'data/test'}
        self.sample_texts = [
            "This is a test sentence.",
            "Another example for testing.",
            "NLP pipeline testing is important."
        ]
        self.sample_labels = [0, 1, 0]
    
    def test_data_ingestion(self):
        """Test data ingestion component."""
        ingestion = DataIngestion(self.config)
        self.assertIsNotNone(ingestion)
        self.assertEqual(ingestion.config, self.config)
    
    def test_text_preprocessing(self):
        """Test text preprocessing component."""
        preprocessor = TextPreprocessor({})
        processed = preprocessor.preprocess(self.sample_texts[0])
        self.assertIsInstance(processed, str)
        self.assertGreater(len(processed), 0)
    
    def test_feature_extraction_tfidf(self):
        """Test TF-IDF feature extraction."""
        extractor = FeatureExtractor({})
        features = extractor.extract_tfidf(self.sample_texts, max_features=10)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
    
    def test_feature_extraction_statistical(self):
        """Test statistical feature extraction."""
        extractor = FeatureExtractor({})
        features = extractor.extract_statistical_features(self.sample_texts)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.sample_texts))
        self.assertIn('text_length', features.columns)
    
    def test_model_training(self):
        """Test model training component."""
        trainer = ModelTrainer({})
        
        # Create simple training data
        X_train = np.random.rand(10, 5)
        y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        X_test = np.random.rand(5, 5)
        y_test = np.array([0, 1, 0, 1, 0])
        
        # Train model
        model = trainer.train(X_train, y_train, model_type='logistic')
        self.assertIsNotNone(model)
        
        # Evaluate
        metrics = trainer.evaluate(X_test, y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIsInstance(metrics['accuracy'], float)

if __name__ == '__main__':
    unittest.main()
