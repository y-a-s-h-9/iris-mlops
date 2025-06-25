# Iris MLOps Project

This project demonstrates a complete end-to-end MLOps pipeline for classifying Iris flowers using a machine learning model, integrating tools like DVC, MLflow, FastAPI, and GitHub Actions.

## Project Features

- ML model training with scikit-learn  
- Dataset versioning using DVC  
- Model tracking and logging with MLflow  
- REST API for inference using FastAPI  
- CI/CD setup via GitHub Actions (optional)  
- Docker-ready for containerization  

## Project Structure

iris-mlops-project/  
├── data/                  # Dataset (iris.csv)  
├── src/                   # Model training and prediction scripts  
│   ├── train.py  
│   ├── predict.py  
│   └── utils.py  
├── app/                   # FastAPI app  
│   └── main.py  
├── models/                # Trained model (model.pkl)  
├── mlruns/                # MLflow logs  
├── .github/workflows/     # GitHub Actions CI/CD  
│   └── ci-cd.yml  
├── requirements.txt  
├── Dockerfile  
├── dvc.yaml  
└── README.md  

## Setup Instructions

1. Clone and Install Dependencies

```bash
git clone <your-repo-url>
cd iris-mlops-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Download the Dataset

```bash
mkdir -p data
curl -o data/iris.csv https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv
```

3. Version the Data (Optional but recommended)

```bash
dvc init
dvc add data/iris.csv
git add .dvc data/.gitignore data/iris.csv.dvc
git commit -m "Add dataset with DVC"
```

4. Train the Model

```bash
python src/train.py
```

Output:
- Trains the model  
- Saves model as models/model.pkl  
- Logs accuracy and model in MLflow  

5. Run the FastAPI App

```bash
uvicorn app.main:app --reload
```

Open in browser:  
http://127.0.0.1:8000  
Swagger docs:  
http://127.0.0.1:8000/docs

6. Make Predictions

Use the Swagger UI to call the /predict endpoint with:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Expected response:

```json
{
  "prediction": "setosa"
}
```

7. Launch MLflow Dashboard

```bash
mlflow ui
```

Open in browser:  
http://localhost:5000

## Tools Used

- scikit-learn for model training  
- joblib for model serialization  
- MLflow for experiment tracking  
- FastAPI for RESTful API  
- DVC for dataset versioning  
- Docker for containerization  
- GitHub Actions for CI/CD  

## For Research Paper

You can include this project in your research to demonstrate:  
- End-to-end machine learning lifecycle  
- Reproducibility and dataset tracking with DVC  
- Model performance tracking with MLflow  
- Real-time predictions using a REST API  
- Automation of retraining and deployment  


