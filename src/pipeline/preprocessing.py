import re
import string
import pandas as pd
import numpy as np
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class TextPreprocessor:
    """Comprehensive text preprocessing for NLP pipeline."""
    
    def __init__(self, config: dict):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """Remove special characters and normalize text."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words from token list."""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stop: bool = True) -> str:
        """Full preprocessing pipeline."""
        text = self.clean_text(text)
        text = self.remove_punctuation(text)
        tokens = self.tokenize(text)
        if remove_stop:
            tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return ' '.join(tokens)
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process entire dataframe."""
        logger.info(f"Processing {len(df)} texts")
        df['processed_text'] = df[text_column].apply(self.preprocess)
        logger.info("Preprocessing complete")
        return df

if __name__ == "__main__":
    # Example usage
    config = {}
    preprocessor = TextPreprocessor(config)
    sample_text = "This is a SAMPLE text with URLs https://example.com!"
    processed = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Processed: {processed}")
