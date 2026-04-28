import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

print("Loading dataset...")

data = pd.read_csv("data/gesture_dataset.csv")

X = data.drop("label", axis=1)
y = data["label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Training model...")

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")

# Save everything
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(encoder, open("model/encoder.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Model + Scaler saved successfully!")