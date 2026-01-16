import pickle

# Load model and vectorizer
with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

# Example passenger data
passenger = {
    'Gender': 'Female',
    'Age': 42,
    'Type of Travel': 'Business travel',
    'Class': 'Eco',
    'Continent': 'Europe',
    'Flight Distance': 1450,
    'Departure Delay in Minutes': 20,
    'Arrival Delay in Minutes': 15
}

# Vectorize input
X = dv.transform([passenger])

# Predict
prediction = model.predict(X)[0]
probability = model.predict_proba(X)[0][1]

# Output result
label = "Satisfied" if prediction == 1 else "Unsatisfied"

print("Prediction:", label)
print("Probability of satisfaction:", round(probability, 3))