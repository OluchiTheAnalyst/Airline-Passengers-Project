from fastapi import FastAPI
import pickle

# Initialize app
app = FastAPI(
    title="HexaWing Passenger Satisfaction API",
    version="1.0"
)

# Load saved model and vectorizer
with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

# Health check endpoint
@app.get("/")
def home():
    return {"message": "HexaWing Satisfaction API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
def predict(passenger: dict):

    # Vectorize input
    X = dv.transform([passenger])

    # Predict class and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    label = "Satisfied" if prediction == 1 else "Unsatisfied"

    return {
        "prediction": label,
        "probability_of_satisfaction": round(float(probability), 3)
    }
