import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix,f1_score,precision_score,recall_score
import joblib

# 1) Load data
df = pd.read_csv(os.path.join("data", "training_data.csv"))

# 2) Clean column names and target
df.columns = df.columns.str.strip()
df['prognosis'] = df['prognosis'].str.strip().str.lower()

# 3) Drop rows with missing target
df = df[df['prognosis'].notna()]

# 4) Split features and label
y = df['prognosis']
X = df.drop(columns=['prognosis'])

# 5) Basic leakage check: drop any feature that correlates almost perfectly with target
le_disease_temp = LabelEncoder().fit(y)
y_enc_temp = le_disease_temp.transform(y)
for col in X.select_dtypes(include=['int64','float64']).columns:
    corr = X[col].corr(pd.Series(y_enc_temp))
    if abs(corr) > 0.99:
        print(f"Dropping leaky feature {col} (corr={corr:.3f})")
        X = X.drop(columns=[col])

# 6) Handle missing values and categorical features
X = X.fillna(0)
X = pd.get_dummies(X, drop_first=True)

# 7) Encode label
le_disease = LabelEncoder()
y_enc = le_disease.fit_transform(y)

# 8) Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

# 9) Define and train a constrained RandomForest to avoid overfitting
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)

# 10) Evaluate on test set
y_pred = model.predict(X_test)
print("Test set accuracy: {:.2%}".format(model.score(X_test, y_test)))
#print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(
    y_test, y_pred, target_names=le_disease.classes_))
    
#print("\nF1 score:", f1_score(y_test, y_pred, average='weighted'))
#print("\nPrecision:", precision_score(y_test, y_pred, average='weighted'))
#print("\nRecall:", recall_score(y_test, y_pred, average='weighted'))

#11) Cross‑validation for more robust estimate
cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy')
print("5-fold CV accuracie:",["{:.2%}".format(s) for s in cv_scores])
print("Mean CV accuracy: {:.2%}".format(cv_scores.mean()))

#12) Save artifacts for deployment
os.makedirs('backend/models', exist_ok=True)
joblib.dump(model, 'backend/models/model_disease.pkl')
joblib.dump(le_disease, 'backend/models/le_disease.pkl')
joblib.dump(X.columns.tolist(), 'backend/models/feature_names.pkl')
with open('backend/models/accuracy.txt','w') as f:
    f.write(f"test_accuracy: {model.score(X_test, y_test):.4f}\n")
    f.write("cv_accuracies: " + ",".join(f"{s:.4f}" for s in cv_scores))
