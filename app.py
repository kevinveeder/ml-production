from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None
feature_names = None

def train_and_save_model():
    """Train a sample model and save it"""
    logger.info("Training new model...")
    
    # Generate sample data (replace with your actual dataset)
    X, y = make_classification(
        n_samples=1000, 
        n_features=10, 
        n_informative=8, 
        n_redundant=2, 
        random_state=42
    )
    
    # Create feature names
    global feature_names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Save model and feature names
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    return model

def load_model():
    """Load the trained model"""
    global model, feature_names
    
    try:
        model = joblib.load('models/model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.info("No saved model found, training new model...")
        model = train_and_save_model()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions on input data"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle single prediction or batch predictions
        if 'features' in data:
            # Single prediction
            features = data['features']
            if len(features) != len(feature_names):
                return jsonify({
                    'error': f'Expected {len(feature_names)} features, got {len(features)}'
                }), 400
            
            # Make prediction
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0].tolist()
            
            return jsonify({
                'prediction': int(prediction),
                'probability': probability,
                'feature_names': feature_names
            })
        
        elif 'batch' in data:
            # Batch predictions
            batch_features = data['batch']
            
            # Validate batch data
            for i, features in enumerate(batch_features):
                if len(features) != len(feature_names):
                    return jsonify({
                        'error': f'Row {i}: Expected {len(feature_names)} features, got {len(features)}'
                    }), 400
            
            # Make predictions
            predictions = model.predict(batch_features).tolist()
            probabilities = model.predict_proba(batch_features).tolist()
            
            return jsonify({
                'predictions': predictions,
                'probabilities': probabilities,
                'count': len(predictions)
            })
        
        else:
            return jsonify({
                'error': 'Invalid request format. Use {"features": [...]} for single prediction or {"batch": [[...], [...]]} for batch'
            }), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'No model loaded'}), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'classes': model.classes_.tolist() if hasattr(model, 'classes_') else None
    })

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model (in production, you'd want authentication here)"""
    try:
        global model
        model = train_and_save_model()
        return jsonify({
            'message': 'Model retrained successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_ENV') == 'development'
    )