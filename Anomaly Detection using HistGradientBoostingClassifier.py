import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib as plt


# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('Dataset01.csv')

# Define features (exclude the 'label' column)
features = data.drop('label', axis=1)

# Standardize features (important for gradient boosting)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, data['label'], test_size=0.2, random_state=42
)

# Create and train the HistGradientBoostingClassifier model
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict anomalies on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


