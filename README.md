## Airline Passenger Satisfaction (DataTalksClub)
### Problem Description
Airline customer satisfaction is influenced by many operational and service-related factors such as delays, seat comfort, onboard service, and travel class. 
However, these factors are often analyzed in isolation, making it difficult for airline staff to understand which variables most strongly impact passenger satisfaction.

This project uses machine learning to analyze passenger-level data from an airline’s database and predict whether a passenger is satisfied or unsatisfied with their travel experience.
The goal is to:
- Identify the most important drivers of passenger satisfaction
- Build predictive models that can estimate satisfaction outcomes
- Provide a deployable service that can be used to support operational and service improvement decisions

The solution can be used by airline operations, customer experience teams, and analysts to better prioritize improvements that have the highest impact on satisfaction.

### Exploratory Data Analysis (EDA)
An extensive EDA was performed to understand the structure and quality of the dataset:
- Checked dataset size, data types, and missing values
- Analyzed numerical feature ranges (age, flight distance, delays)
- Examined class distribution of the target variable (Satisfied)
- Investigated relationships between satisfaction and:
  - Type of travel (business vs personal)
  - Travel class
  - Delay times
  - Service quality ratings
- Identified class imbalance and assessed its impact on modeling
- Performed feature importance analysis using mutual information

Key insights from EDA were used to guide feature selection and model choice.

### Model Training
Multiple models were trained and compared:
- Logistic Regression (baseline linear model)
- Decision Tree
- Random Forest
- XGBoost (Gradient Boosting)

Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

Hyperparameter tuning was performed using RandomizedSearchCV with stratified cross-validation for tree-based models.

The final selected model achieved strong predictive performance while maintaining good recall for the minority class.

### Exporting Notebook to Script
The logic for training the model was exported from the notebook into a standalone script:
- `train.py`
- Loads data
- Performs train/validation/test split
- Applies feature encoding
- Trains the final model
- Saves the trained model and vectorizer to disk

This ensures the model can be trained without relying on the notebook.

### Reproducibility
The project is fully reproducible:
- All dependencies are listed in `requirements.txt`
- Training can be reproduced by running `train.py`
- The dataset is included in the repository (or clear instructions are provided)
- No manual notebook execution is required to retrain the model

### Model Deployment
A prediction service was built using FastAPI:
- `serve.py` exposes a REST API with a `/predict` endpoint
- The API accepts passenger features as JSON input
- Returns a satisfaction prediction (`Satisfied` / `Unsatisfied`)

This allows the trained model to be used as a service by other systems.

### Dependency and Environment Management
- Dependencies are managed using `requirements.txt`
- A Python virtual environment was used during development
- The README provides instructions on how to install dependencies and run the project

### Containerization
The application is fully containerized using Docker.
- A `Dockerfile` is provided
- The container installs dependencies, loads the model, and runs the API

#### Build the Docker image
```
docker build -t airline-satisfaction .
```
#### Run the container
```
docker run -p 8000:8000 airline-satisfaction
```
The API will be available at:
```
http://localhost:8000
```

### Cloud Deployment
Cloud deployment was not implemented in this project.
However, the containerized service can be easily deployed to platforms such as:
- AWS ECS
- Google Cloud Run
- Azure Container Apps
- Kubernetes
