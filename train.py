import pandas as pd 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np
from sklearn.feature_extraction import DictVectorizer


# 1. Load dataset
df = pd.read_csv('hexawing_data.csv', encoding='latin-1')

target = 'Satisfied'
features =['Gender', 'Age', 'Type of Travel', 'Class', 'Continent', 'Flight Distance','Departure Delay in Minutes', 'Arrival Delay in Minutes']

X = df[features]
y = df[target].map({'Y': 1, 'N': 0})

# First split: train vs test
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Second split: train vs validation
X_train, X_val, y_train, y_val = train_test_split(
    X_full_train,
    y_full_train,
    test_size=0.25,
    stratify=y_full_train,
    random_state=42
)

# One-hot encode categorical features
dv = DictVectorizer(sparse=True)

train_dicts = X_train.to_dict(orient='records')
val_dicts   = X_val.to_dict(orient='records')

X_train_vec = dv.fit_transform(train_dicts)
X_val_vec   = dv.transform(val_dicts)

# 5. Train XGBoost model (using your tuned parameters)
model = XGBClassifier(
    n_estimators=200,       # number of trees
    max_depth=7,           # depth of each tree
    learning_rate=0.01,      # step size shrinkage
    subsample=0.6,          # sample ratio of training instance
    gamma=0,
    colsample_bytree=0.8,   # subsample ratio of columns when constructing each tree
    min_child_weight = 3,
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss', # for multi-class classification
)

model.fit(X_train_vec, y_train)

#6. Save model
with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Model trained and saved as model.bin")