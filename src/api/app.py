from flask import Flask, request, jsonify
import joblib
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'NLP API is running'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        # Transform text using vectorizer
        if vectorizer is None:
            return jsonify({'error': 'Vectorizer not loaded'}), 500
        
        features = vectorizer.transform([text])
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        response = {
            'text': text,
            'prediction': int(prediction[0]),
            'confidence': float(prediction_proba[0][prediction[0]]) if prediction_proba is not None else None
        }
        
        logger.info(f"Prediction made: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'texts must be a list'}), 400
        
        # Transform texts
        features = vectorizer.transform(texts)
        
        # Make predictions
        predictions = model.predict(features)
        predictions_proba = model.predict_proba(features) if hasattr(model, 'predict_proba') else None
        
        results = []
        for i, text in enumerate(texts):
            result = {
                'text': text,
                'prediction': int(predictions[i]),
                'confidence': float(predictions_proba[i][predictions[i]]) if predictions_proba is not None else None
            }
            results.append(result)
        
        logger.info(f"Batch prediction made for {len(texts)} texts")
        return jsonify({'results': results}), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def load_models(model_path: str, vectorizer_path: str):
    """Load trained model and vectorizer."""
    global model, vectorizer
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

if __name__ == '__main__':
    # Load models if paths are provided
    # load_models('models/model.pkl', 'models/vectorizer.pkl')
    app.run(host='0.0.0.0', port=5000, debug=True)
