# 🫀 Heart Disease Prediction API

Production-ready ML API that predicts heart disease probability using LightGBM.
Built with FastAPI, tracked with MLflow, and containerized with Docker.

---

## 🏗️ Architecture

Heart Disease Dataset → Feature Engineering → LightGBM Training → MLflow Tracking → FastAPI → Docker

| Component               | Tool                         | Description                                 |
|-------------------------|------------------------------|---------------------------------------------|
| **Data**                | Kaggle Heart Disease Dataset | 1,025 patients, 13 clinical features        |
| **Feature Engineering** | Python + pandas              | 3 additional features, StandardScaler       |
| **Model**               | LightGBM                     | Gradient boosting classifier                |
| **Experiment Tracking** | MLflow                       | 3 experiments compared, best model selected |
| **API**                 | FastAPI                      | REST endpoint with automatic documentation  |
| **Containerization**    | Docker                       | Portable, reproducible deployment           |

---

## 📊 Model Performance

| Experiment      | Accuracy   | F1 Score   | ROC AUC    |
|-----------------|------------|------------|------------|
| Base params     | **97.07%** | **0.9717** | **0.9987** |
| More estimators | 92.68%     | 0.9289     | 0.9858     |
| Regularization  | 96.10%     | 0.9623     | 0.9900     |

Best model: Experiment 1 selected for deployment.

---

## 🛠️ Tech Stack

| Tool         | Version | Purpose             |
|--------------|---------|---------------------|
| Python       | 3.11    | Core language       |
| LightGBM     | 4.x     | ML model            |
| MLflow       | 2.x     | Experiment tracking |
| FastAPI      | 0.x     | REST API            |
| Docker       | 29.x    | Containerization    |
| scikit-learn | 1.x     | Preprocessing       |

---

## 📁 Project Structure

    ml-prediction-api/
    ├── data/
    │   └── heart.csv              # Dataset (not tracked in git)
    ├── src/
    │   ├── features/
    │   │   ├── explore.py         # Data exploration
    │   │   └── build_features.py  # Feature engineering
    │   ├── models/
    │   │   ├── train.py           # MLflow experiment tracking
    │   │   ├── save_model.py      # Save best model
    │   │   ├── model.pkl          # Trained model
    │   │   └── scaler.pkl         # Feature scaler
    │   └── api/
    │       └── main.py            # FastAPI application
    ├── utils/
    │   └── logger.py              # Centralized logging
    ├── Dockerfile
    └── requirements.txt

---

## 🚀 How to Run

### Option A — Local

```bash
git clone git@github.com:DylanRReexx/ml-prediction-api.git
cd ml-prediction-api
python -m venv venv
venv\Scripts\Activate
pip install -r requirements.txt
python src/models/save_model.py
uvicorn src.api.main:app --reload
```

### Option B — Docker

```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
```

API available at: **http://localhost:8000**
Documentation at: **http://localhost:8000/docs**

---

## 🔍 API Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | API status               |
| GET    | `/health`  | Health check             |
| POST   | `/predict` | Heart disease prediction |

### Example Request

```json
{
  "age": 55,
  "sex": 1,
  "cp": 0,
  "trestbps": 150,
  "chol": 280,
  "fbs": 1,
  "restecg": 1,
  "thalach": 130,
  "exang": 1,
  "oldpeak": 2.5,
  "slope": 1,
  "ca": 2,
  "thal": 2
}
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.8731,
  "diagnosis": "Heart Disease Detected",
  "risk_level": "High"
}
```

---

## 🧪 Data Quality

- No null values in dataset
- Target variable validated (binary 0/1)
- Feature ranges validated before prediction
- Input validation via Pydantic models

---

## 📐 How to Scale

| Current          | Production Scale                   |
|------------------|------------------------------------|
| Local Docker     | Kubernetes or AWS ECS              |
| Manual training  | Automated retraining pipeline      |
| MLflow local     | MLflow on cloud (Databricks, AWS)  |
| Single container | Load balancer + multiple replicas  |
| pickle model     | Model registry (MLflow, SageMaker) |

---

## 👤 Author

**Dylan** — Systems Engineering Student @ ULATINA
[GitHub](https://github.com/DylanRReexx)