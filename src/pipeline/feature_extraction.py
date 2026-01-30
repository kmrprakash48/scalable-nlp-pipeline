import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim.models import Word2Vec
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureExtractor:
    """Extract features from preprocessed text."""
    
    def __init__(self, config: dict):
        self.config = config
        self.vectorizer = None
        self.word2vec_model = None
        logger.info("FeatureExtractor initialized")
    
    def extract_tfidf(self, texts: List[str], max_features: int = 5000) -> np.ndarray:
        """Extract TF-IDF features."""
        logger.info("Extracting TF-IDF features")
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        features = self.vectorizer.fit_transform(texts)
        logger.info(f"Extracted {features.shape[1]} TF-IDF features")
        return features.toarray()
    
    def extract_count_vectors(self, texts: List[str], max_features: int = 5000) -> np.ndarray:
        """Extract count-based features."""
        logger.info("Extracting count vectors")
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2
        )
        features = self.vectorizer.fit_transform(texts)
        logger.info(f"Extracted {features.shape[1]} count features")
        return features.toarray()
    
    def train_word2vec(self, tokenized_texts: List[List[str]], 
                       vector_size: int = 100, 
                       window: int = 5,
                       min_count: int = 2) -> Word2Vec:
        """Train Word2Vec embeddings."""
        logger.info("Training Word2Vec model")
        self.word2vec_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        logger.info("Word2Vec training complete")
        return self.word2vec_model
    
    def get_word2vec_features(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """Get average word2vec embeddings for texts."""
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not trained. Call train_word2vec first.")
        
        features = []
        for tokens in tokenized_texts:
            vectors = [self.word2vec_model.wv[token] 
                      for token in tokens 
                      if token in self.word2vec_model.wv]
            if vectors:
                features.append(np.mean(vectors, axis=0))
            else:
                features.append(np.zeros(self.word2vec_model.vector_size))
        
        logger.info(f"Generated {len(features)} word2vec feature vectors")
        return np.array(features)
    
    def extract_statistical_features(self, texts: List[str]) -> pd.DataFrame:
        """Extract statistical text features."""
        logger.info("Extracting statistical features")
        features = pd.DataFrame({
            'text_length': [len(text) for text in texts],
            'word_count': [len(text.split()) for text in texts],
            'avg_word_length': [np.mean([len(word) for word in text.split()]) if text.split() else 0 
                               for text in texts],
            'unique_words': [len(set(text.split())) for text in texts]
        })
        logger.info("Statistical features extracted")
        return features

if __name__ == "__main__":
    # Example usage
    config = {}
    extractor = FeatureExtractor(config)
    sample_texts = ["sample text one", "another sample text"]
    tfidf_features = extractor.extract_tfidf(sample_texts)
    print(f"TF-IDF shape: {tfidf_features.shape}")
