# Scalable NLP Data Pipeline

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

A production-ready, scalable Natural Language Processing (NLP) data pipeline with end-to-end functionality including data ingestion, preprocessing, feature extraction, model training, and deployment.

## 🎯 Features

- **Modular Architecture**: Clean separation of concerns with independent pipeline components
- **Scalability**: Horizontal scaling with Celery, Redis, and containerization
- **Production-Ready**: FastAPI-based REST API with monitoring and logging
- **Flexibility**: Configurable pipeline stages through YAML configuration
- **Multiple NLP Models**: Support for transformers, LSTM, CNN, and traditional ML models
- **Comprehensive Testing**: Unit tests, integration tests with pytest
- **CI/CD Ready**: GitHub Actions workflow included
- **Docker Support**: Containerized deployment with Docker Compose
- **Monitoring**: Prometheus metrics and structured logging

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Testing](#testing)
- [Contributing](#contributing)

## 📁 Project Structure

```
scalable-nlp-pipeline/
├── config/
│   └── config.yaml           # Pipeline configuration
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py    # Data loading and ingestion
│   │   ├── preprocessing.py     # Text cleaning and preprocessing
│   │   ├── feature_extraction.py# Feature engineering
│   │   ├── model_training.py    # Model training logic
│   │   └── orchestrator.py      # Pipeline orchestration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py            # FastAPI application
│   │   └── endpoints.py         # API endpoints
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging configuration
│       └── monitoring.py        # Metrics and monitoring
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml               # CI/CD pipeline
├── data/
│   ├── raw/                     # Raw data storage
│   ├── processed/               # Processed data
│   └── models/                  # Trained models
├── notebooks/
│   └── example_usage.ipynb      # Example notebooks
├── logs/
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- Redis (for task queue)
- PostgreSQL (optional, for metadata storage)

### Local Installation

1. **Clone the repository**

```bash
git clone https://github.com/kmrprakash48/scalable-nlp-pipeline.git
cd scalable-nlp-pipeline
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install the package**

```bash
pip install -e .
```

5. **Download spaCy model**

```bash
python -m spacy download en_core_web_sm
```

### Docker Installation

```bash
docker-compose up --build
```

## ⚡ Quick Start

### 1. Configure the Pipeline

Edit `config/config.yaml` to customize pipeline settings:

```yaml
pipeline:
  name: "my-nlp-pipeline"
  version: "1.0.0"

ingestion:
  batch_size: 1000
  max_workers: 4

preprocessing:
  text_cleaning:
    lowercase: true
    remove_stopwords: true
```

### 2. Run the Pipeline

```python
from pipeline.orchestrator import NLPPipeline

# Initialize pipeline
pipeline = NLPPipeline(config_path="config/config.yaml")

# Run end-to-end pipeline
results = pipeline.run(
    input_data="data/raw/text_data.csv",
    output_dir="data/processed"
)
```

### 3. Start the API Server

```bash
nlp-api
# or
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`

### 4. Make API Requests

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text for classification"}'
```

## ⚙️ Configuration

The pipeline is highly configurable through `config/config.yaml`:

- **Data Ingestion**: Configure data sources (API, files, database)
- **Preprocessing**: Text cleaning, tokenization, normalization options
- **Feature Extraction**: TF-IDF, Word2Vec, BERT embeddings
- **Model Training**: Hyperparameters, validation split, early stopping
- **Deployment**: API settings, scaling parameters
- **Monitoring**: Logging levels, metrics collection

## 📖 Usage

### Data Ingestion

```python
from pipeline.data_ingestion import DataIngester

ingester = DataIngester(config)
data = ingester.load_from_csv("data/raw/dataset.csv")
# or
data = ingester.load_from_api("https://api.example.com/data")
```

### Preprocessing

```python
from pipeline.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(config)
cleaned_data = preprocessor.clean_text(data)
tokens = preprocessor.tokenize(cleaned_data)
```

### Feature Extraction

```python
from pipeline.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(config)
features = extractor.extract_tfidf(tokens)
# or
features = extractor.extract_bert_embeddings(tokens)
```

### Model Training

```python
from pipeline.model_training import ModelTrainer

trainer = ModelTrainer(config)
model = trainer.train(
    features=features,
    labels=labels,
    model_type="transformer"
)
trainer.save_model(model, "data/models/nlp_model.pt")
```

## 🔌 API Documentation

### Endpoints

#### Health Check
```
GET /health
```

#### Predict
```
POST /api/v1/predict
Body: {"text": "string"}
Response: {"prediction": "label", "confidence": 0.95}
```

#### Batch Predict
```
POST /api/v1/batch-predict
Body: {"texts": ["text1", "text2"]}
Response: {"predictions": [{...}, {...}]}
```

#### Train Model
```
POST /api/v1/train
Body: {"data_path": "string", "model_config": {}}
```

## 🐳 Deployment

### Docker Deployment

1. **Build and run with Docker Compose**

```bash
docker-compose up -d
```

2. **Scale services**

```bash
docker-compose up -d --scale api=3
```

### Kubernetes Deployment

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up SSL/TLS certificates
- [ ] Configure database connections
- [ ] Set up Redis for caching
- [ ] Configure monitoring and alerts
- [ ] Set up log aggregation
- [ ] Configure auto-scaling
- [ ] Set up backup strategies

## 🧪 Testing

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=src tests/
```

### Run specific test suite

```bash
pytest tests/unit/
pytest tests/integration/
```

## 📊 Monitoring

The pipeline includes built-in monitoring with Prometheus metrics:

- Request count and latency
- Model inference time
- Pipeline stage execution time
- Error rates and types
- System resource usage

Access Prometheus metrics at `http://localhost:9090/metrics`

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers
- spaCy
- FastAPI
- Scikit-learn

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This is a production-ready template. Customize the configuration, models, and features according to your specific use case.
