import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
df = pd.read_csv(r'C:\Users\VICTUS\ARPAN DOC\archive (3)\IRIS.csv')

# 2. Preprocessing
# Using actual column names from your dataset
X = df.drop('species', axis=1)
y = df['species']

# 3. Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Additional info
print(f"\nDataset shape: {df.shape}")
print(f"Classes: {df['species'].unique()}")
