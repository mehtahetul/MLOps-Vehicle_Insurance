# Vehicle Insurance Cross-Sell Prediction — End-to-End MLOps Project

Predicts whether an existing health-insurance customer will be interested in buying **vehicle insurance**, and wraps the model in a production-style MLOps pipeline: data pulled from **MongoDB Atlas**, a modular training pipeline, model versioning in **AWS S3**, a **FastAPI** web app for predictions, and container-based deployment to **AWS EC2** through **Docker**, **ECR** and **GitHub Actions**.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Prediction Pipeline & Web App](#prediction-pipeline--web-app)
- [Tech Stack](#tech-stack)
- [Getting Started (Local)](#getting-started-local)
- [Deployment on AWS](#deployment-on-aws)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Problem Statement

An insurance company wants to know which of its existing customers are likely to respond positively to a vehicle insurance offer. This is a **binary classification** problem (`Response` = 1 → interested, 0 → not interested) on a heavily **imbalanced** dataset, where only a small share of customers respond positively.

## Dataset

~381,000 customer records with the following columns:

| Column | Description |
|---|---|
| `id` | Unique customer ID (dropped before training) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Driving_License` | 1 if the customer holds a driving licence |
| `Region_Code` | Code of the customer's region |
| `Previously_Insured` | 1 if the customer already has vehicle insurance |
| `Vehicle_Age` | `< 1 Year`, `1-2 Year`, `> 2 Years` |
| `Vehicle_Damage` | Yes / No — vehicle damaged in the past |
| `Annual_Premium` | Premium paid per year |
| `Policy_Sales_Channel` | Anonymised outreach channel code |
| `Vintage` | Days the customer has been with the company |
| `Response` | **Target** — 1 if interested, 0 otherwise |

The raw CSV is uploaded once to a MongoDB Atlas collection (see `notebook/mongoDB_demo.ipynb`); the pipeline always reads from MongoDB, not from the CSV.

## Architecture

```
                ┌──────────────────────── Training Pipeline ────────────────────────┐
MongoDB Atlas → │ Ingestion → Validation → Transformation → Trainer → Evaluation → Pusher │ → AWS S3 (model registry)
                └────────────────────────────────────────────────────────────────────┘
                                                                                          │
User ──(HTML form)──► FastAPI app (app.py) ──► Prediction Pipeline ──► loads model from S3 ┘
                              ▲
         GitHub push ──► GitHub Actions ──► Docker image ──► AWS ECR ──► EC2 (self-hosted runner)
```

## Project Structure

```
.
├── app.py                      # FastAPI app: form UI, /train and prediction routes
├── demo.py                     # Runs the full training pipeline from the CLI
├── template.py                 # Scaffolds the folder structure (run once at project start)
├── config/
│   ├── schema.yaml             # Expected columns/dtypes + feature groups for scaling
│   └── model.yaml              # Reserved for model hyperparameters
├── notebook/
│   ├── exp-notebook.ipynb      # EDA, preprocessing and model experiments
│   └── mongoDB_demo.ipynb      # Uploads the dataset to MongoDB Atlas
├── src/
│   ├── components/             # One class per pipeline stage
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/          # MongoDB and AWS (boto3) clients
│   ├── cloud_storage/          # S3 helper: upload / download / check model files
│   ├── data_access/            # Reads a MongoDB collection into a DataFrame
│   ├── constants/              # All project-wide constants (paths, names, thresholds)
│   ├── entity/
│   │   ├── config_entity.py    # Config dataclasses for each stage
│   │   ├── artifact_entity.py  # Output (artifact) dataclasses for each stage
│   │   ├── estimator.py        # MyModel = preprocessing object + trained model
│   │   └── s3_estimator.py     # Loads / saves / predicts with the model stored in S3
│   ├── pipline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── logger/                 # Rotating file + console logging (logs/)
│   ├── exception/              # Custom exception with file name and line number
│   └── utils/main_utils.py     # YAML, pickle (dill) and NumPy save/load helpers
├── templates/vehicledata.html  # Input form
├── static/css/style.css
├── Dockerfile
├── requirements.txt
├── setup.py / pyproject.toml   # Makes `src` installable as a local package (`-e .`)
└── LICENSE
```

## Training Pipeline

Each stage takes a **config** object and the previous stage's **artifact**, and returns its own artifact. Every run writes its outputs to a timestamped folder under `artifact/`.

1. **Data Ingestion** — connects to MongoDB using `MONGODB_URL`, exports the collection to a feature-store CSV, and splits it 75/25 into `train.csv` and `test.csv`.
2. **Data Validation** — checks the column count and the presence of every numerical and categorical column defined in `config/schema.yaml`; writes a `report.yaml` with the result.
3. **Data Transformation**
   - Maps `Gender` to 0/1, drops `_id`, one-hot encodes `Vehicle_Age` and `Vehicle_Damage`
   - `StandardScaler` on `Age`, `Vintage`; `MinMaxScaler` on `Annual_Premium`
   - Balances classes with **SMOTEENN**
   - Saves the fitted preprocessor (`preprocessing.pkl`) and transformed NumPy arrays
4. **Model Trainer** — trains a `RandomForestClassifier`
   (`n_estimators=200`, `max_depth=10`, `min_samples_split=7`, `min_samples_leaf=6`, `criterion='entropy'`), computes F1 / precision / recall, and saves a single `MyModel` object that bundles the preprocessor with the model.
5. **Model Evaluation** — compares the new model's F1 score with the model currently in S3 (if any). The new model is accepted only if it scores higher.
6. **Model Pusher** — uploads the accepted model to the S3 bucket under `model-registry/model.pkl`.

## Prediction Pipeline & Web App

`app.py` serves a FastAPI application:

| Route | Method | What it does |
|---|---|---|
| `/` | GET | Renders the input form |
| `/` | POST | Reads the form, builds a DataFrame, loads the latest model from S3 and shows **Response-Yes** / **Response-No** |
| `/train` | GET | Runs the full training pipeline |

The form expects already-encoded values (e.g. `Gender` 1/0, `Vehicle_Age_lt_1_Year` 1/0, `Vehicle_Damage_Yes` 1/0).

## Tech Stack

- **Language:** Python 3.10
- **ML:** pandas, NumPy, scikit-learn, imbalanced-learn (SMOTEENN)
- **Data store:** MongoDB Atlas (pymongo)
- **Model registry:** AWS S3 (boto3)
- **Serving:** FastAPI, Uvicorn, Jinja2
- **DevOps:** Docker, AWS ECR, AWS EC2, GitHub Actions (self-hosted runner)

## Getting Started (Local)

### 1. Clone and create an environment
```bash
git clone https://github.com/mehtahetul/MLOps-Vehicle_Insurance.git
cd MLOps-Vehicle_Insurance

python -m venv vehicle
source vehicle/bin/activate        # Windows: vehicle\Scripts\activate
pip install -r requirements.txt    # also installs src/ as a local package via "-e ."
```

### 2. Set up MongoDB Atlas
1. Create a free (M0) cluster and a database user.
2. Under **Network Access**, allow your IP (or `0.0.0.0/0` for testing).
3. Copy the Python driver connection string.
4. Run `notebook/mongoDB_demo.ipynb` to upload the dataset to database `Proj1`, collection `Proj1-Data`. Read the connection string from an environment variable rather than writing it into the notebook.

### 3. Set up AWS S3
1. Create an IAM user with S3 access and generate an access key.
2. Create an S3 bucket in `us-east-1` and set `MODEL_BUCKET_NAME` in `src/constants/__init__.py` to its name.

### 4. Export environment variables
```bash
# bash / zsh
export MONGODB_URL="mongodb+srv://<user>:<password>@<cluster>/?appName=Cluster0"
export AWS_ACCESS_KEY_ID="<your-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<your-secret-access-key>"
```
```powershell
# PowerShell
$env:MONGODB_URL = "mongodb+srv://<user>:<password>@<cluster>/?appName=Cluster0"
$env:AWS_ACCESS_KEY_ID = "<your-access-key-id>"
$env:AWS_SECRET_ACCESS_KEY = "<your-secret-access-key>"
```

### 5. Train
```bash
python demo.py
```
Outputs go to `artifact/<timestamp>/` and logs to `logs/`. If the model is accepted it is pushed to S3.

### 6. Run the app
```bash
python app.py
```
Open http://localhost:5000 to make predictions, or http://localhost:5000/train to retrain.

## Deployment on AWS

1. **ECR** — create a repository (e.g. `vehicleproj`) to hold the Docker image.
2. **EC2** — launch an Ubuntu instance, install Docker, and open the app port (5000) in the security group's inbound rules.
3. **Self-hosted runner** — in GitHub: *Settings → Actions → Runners → New self-hosted runner*, then run the given commands on the EC2 instance.
4. **GitHub secrets** — add `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `ECR_REPO` (plus `MONGODB_URL` for training).
5. **CI/CD** — a GitHub Actions workflow in `.github/workflows/` builds the image, pushes it to ECR, and pulls and runs it on EC2 on every push to `main`.
6. Visit `http://<EC2-public-IP>:5000`.

To build and run the container yourself:
```bash
docker build -t vehicle-insurance .
docker run -p 5000:5000 \
  -e MONGODB_URL -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  vehicle-insurance
```

## Configuration Reference

Key settings in `src/constants/__init__.py`:

| Constant | Default | Purpose |
|---|---|---|
| `DATABASE_NAME` / `COLLECTION_NAME` | `Proj1` / `Proj1-Data` | MongoDB source |
| `DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO` | `0.25` | Test split size |
| `MODEL_TRAINER_EXPECTED_SCORE` | `0.6` | Minimum accuracy for a trained model |
| `MODEL_BUCKET_NAME` | — | S3 bucket used as the model registry |
| `MODEL_PUSHER_S3_KEY` | `model-registry` | S3 prefix for the deployed model |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `5000` | FastAPI server |

## License

Released under the [MIT License](LICENSE).
