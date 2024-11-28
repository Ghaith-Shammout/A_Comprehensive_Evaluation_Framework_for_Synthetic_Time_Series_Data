from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.hybrid import HIVECOTEV2

# Load your energy consumption dataset
# Assume X_real and y_real are your features and target arrays (preprocessed time series)
# Replace this with your actual dataset loading process
X_real = ...  # Replace with feature array
y_real = ...  # Replace with target array

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)

# Function to train and evaluate a classifier
def train_and_evaluate_model(clf, model_name):
    print(f"Training {model_name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='binary')
    print(f"{model_name} F1-Score: {f1:.2f}")
    return f1

# 1. TimeSeriesForestClassifier
tsf_clf = TimeSeriesForestClassifier(n_estimators=100, random_state=42)
tsf_f1 = train_and_evaluate_model(tsf_clf, "TimeSeriesForestClassifier")

# 2. ShapeletTransformClassifier
shapelet_clf = ShapeletTransformClassifier(estimator="randomforest", random_state=42)
shapelet_f1 = train_and_evaluate_model(shapelet_clf, "ShapeletTransformClassifier")

# 3. HybridIntervalTree (HIVECOTE)
hivecote_clf = HIVECOTEV2(random_state=42)
hivecote_f1 = train_and_evaluate_model(hivecote_clf, "HybridIntervalTree (HIVECOTE)")

# Compare results
print("\nModel Performance Comparison:")
print(f"TimeSeriesForestClassifier F1-Score: {tsf_f1:.2f}")
print(f"ShapeletTransformClassifier F1-Score: {shapelet_f1:.2f}")
print(f"HybridIntervalTree (HIVECOTE) F1-Score: {hivecote_f1:.2f}")
