"""Scalable NLP Pipeline Package."""

__version__ = '1.0.0'
__author__ = 'Your Name'

from src.pipeline.data_ingestion import DataIngestion
from src.pipeline.preprocessing import TextPreprocessor
from src.pipeline.feature_extraction import FeatureExtractor
from src.pipeline.model_training import ModelTrainer
from src.utils.logger import setup_logger

__all__ = [
    'DataIngestion',
    'TextPreprocessor',
    'FeatureExtractor',
    'ModelTrainer',
    'setup_logger'
]
