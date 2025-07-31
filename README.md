# Deployable ML Prediction Service *Template*
This project is a flask-based ml production service boilerplate, containerized with docker, designed for rapid deployment. Just add your data and train a model

## Features

- RESTful API for ML predictions
- Single and batch prediction support
- Model health checks and information endpoints
- Docker containerization
- Error handlinng and logging
- Model retraining endpoint
- Production-ready with Gunicorn

## Quick Start

### Local Development

1. **Clone and setup**
   ```bash
   git clone <your-repo>
   cd ml-prediction-service
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the service**
   ```bash
   python app.py
   ```

3. **Test the API**
   ```bash
   python test_client.py
   ```

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t ml-prediction-service .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 ml-prediction-service
   ```

## API Endpoints

### Health Check
```http
GET /health
```
Returns service health status and model availability.

### Model Information
```http
GET /model-info
```
Returns information about the loaded model including feature names and model type.

### Single Prediction
```http
POST /predict
Content-Type: application/json

{
  "features": [1.2, -0.5, 0.8, 1.1, -1.3, 0.4, -0.7, 0.9, 1.5, -0.2]
}
```

### Batch Predictions
```http
POST /predict
Content-Type: application/json

{
  "batch": [
    [1.2, -0.5, 0.8, 1.1, -1.3, 0.4, -0.7, 0.9, 1.5, -0.2],
    [0.3, 1.1, -0.4, 0.7, 0.9, -1.2, 0.6, -0.8, 0.2, 1.4]
  ]
}
```

### Retrain Model
```http
POST /retrain
```
Retrains the model with new data (implement authentication in production).

## Response Examples

### Single Prediction Response
```json
{
  "prediction": 1,
  "probability": [0.3, 0.7],
  "feature_names": ["feature_0", "feature_1", ..., "feature_9"]
}
```

### Batch Prediction Response
```json
{
  "predictions": [1, 0, 1],
  "probabilities": [[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]],
  "count": 3
}
```

## Customization

### Using Your Own Model

1. **Replace the training data** in `train_and_save_model()`:
   ```python
   # Replace this section with your data loading
   X, y = make_classification(...)  # Remove this line
   
   # Add your data loading logic
   df = pd.read_csv('your_data.csv')
   X = df.drop('target', axis=1).values
   y = df['target'].values
   ```

2. **Update feature names** to match your dataset:
   ```python
   feature_names = df.drop('target', axis=1).columns.tolist()
   ```

3. **Choose your model**:
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.svm import SVC
   from sklearn.ensemble import GradientBoostingClassifier
   
   model = LogisticRegression()  # or any other model
   ```

### Environment Variables

- `PORT`: Service port (default: 5000)
- `FLASK_ENV`: Set to 'development' for debug mode

## Production Considerations

- Add authentication to sensitive endpoints like `/retrain`
- Implement rate limiting
- Add model versioning
- Set up monitoring and alerting
- Use a proper database for model metadata
- Implement A/B testing for model versions
- Add input validation and sanitization
- Set up logging aggregation

## Testing

The `test_client.py` script provides comprehensive API testing:

```bash
python test_client.py
```

## Deployment Options

- **Docker**: Use the provided Dockerfile
- **Cloud Run**: Works out of the box with the Dockerfile
- **Kubernetes**: Create deployment and service manifests
- **Heroku**: Add a `Procfile` with the gunicorn command
- **AWS ECS/EC2**: Use the Docker image

## Monitoring

The service includes:
- Health check endpoint for load balancers
- Structured logging
- Error handling with appropriate HTTP status codes

Add monitoring tools like Prometheus, New Relic, or DataDog for production use.
