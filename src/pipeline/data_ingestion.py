import os
import json
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import requests
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataIngestion:
    """Handle data ingestion from multiple sources."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/raw'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataIngestion initialized")
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_from_json(self, file_path: str) -> List[Dict]:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            raise
    
    def load_from_api(self, url: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Load data from API endpoint."""
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} records from API")
            return df
        except Exception as e:
            logger.error(f"Error loading from API: {e}")
            raise
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """Save data to disk."""
        try:
            output_path = self.data_dir / filename
            data.to_csv(output_path, index=False)
            logger.info(f"Saved data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    config = {'data_dir': 'data/raw'}
    ingestion = DataIngestion(config)
    print("Data ingestion module ready")
