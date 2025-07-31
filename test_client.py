import requests
import json
import numpy as np

# Base URL - change this to your deployed service URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_model_info():
    """Test model info endpoint"""
    print("Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_single_prediction():
    """Test single prediction"""
    print("Testing single prediction...")
    
    # Generate random features (10 features as per the model)
    features = np.random.randn(10).tolist()
    
    data = {
        "features": features
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_batch_prediction():
    """Test batch prediction"""
    print("Testing batch prediction...")
    
    # Generate random batch of features
    batch_features = [np.random.randn(10).tolist() for _ in range(3)]
    
    data = {
        "batch": batch_features
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Request: {json.dumps(data, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    # Test with wrong number of features
    data = {
        "features": [1, 2, 3]  # Should be 10 features
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("ML Prediction Service API Tests")
    print("=" * 40)
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_error_handling()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the service. Make sure it's running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")