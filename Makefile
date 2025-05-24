# Variables 
VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip
UVICORN = $(VENV)/Scripts/uvicorn

DATA_PATH = data/survey_lung_cancer.csv
MODEL_PATH = models/SVM_model.pkl
FEATURES_PATH = models/feature_names.pkl
FEATURES = [1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2]

# Docker
IMAGE_NAME = tesnimmt/cancer-fastapi
TAG = latest

.PHONY: init
init:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: run
run:
	$(UVICORN) run:app --reload

.PHONY: train
train:
	curl -X POST http://127.0.0.1:8000/api/train \
	-H "Content-Type: application/json" \
	-d "{\"data_path\": \"$(DATA_PATH)\", \"model_save_path\": \"$(MODEL_PATH)\"}"

.PHONY: predict
predict:
	curl -X POST http://127.0.0.1:8000/api/predict \
	-H "Content-Type: application/json" \
	-d "{\"model_path\": \"$(MODEL_PATH)\", \"features\": $(FEATURES)}"

.PHONY: clean
clean:
	powershell -Command "if (Test-Path '__pycache__') { Remove-Item '__pycache__' -Recurse -Force }"
	powershell -Command "if (Test-Path 'venv') { Remove-Item 'venv' -Recurse -Force }"

# Docker commands
.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE_NAME):$(TAG) .

.PHONY: docker-run
docker-run:
	docker run -d -p 8000:8000 $(IMAGE_NAME):$(TAG)

.PHONY: docker-push
docker-push:
	docker push $(IMAGE_NAME):$(TAG)

.PHONY: docker-clean
docker-clean:
	docker rmi $(IMAGE_NAME):$(TAG)

.PHONY: all
all: docker-build docker-run

.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000