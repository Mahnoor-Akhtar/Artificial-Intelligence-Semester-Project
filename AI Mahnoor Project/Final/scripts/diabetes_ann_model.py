"""
Diabetes Prediction using Artificial Neural Network (ANN)
Framework: TensorFlow/Keras
Model Type: Binary Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 70)
print("DIABETES PREDICTION - ARTIFICIAL NEURAL NETWORK")
print("=" * 70)

# ========================
# 1. LOAD DATASET
# ========================
print("\n[1] Loading Dataset...")
df = pd.read_csv('../data/diabetes_dataset.csv')
print(f"Dataset Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nTarget Distribution:\n{df['diabetes'].value_counts()}")

# ========================
# 2. DATA PREPROCESSING
# ========================
print("\n[2] Data Preprocessing...")

# Handle categorical variables
print("Encoding categorical variables...")
label_encoders = {}
categorical_columns = ['gender', 'smoking_history']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  - {col}: {le.classes_}")

# Separate features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Feature Scaling using StandardScaler
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling completed!")
print(f"Mean of scaled training data: {X_train_scaled.mean():.6f}")
print(f"Std of scaled training data: {X_train_scaled.std():.6f}")

# ========================
# 3. BUILD ANN MODEL
# ========================
print("\n[3] Building ANN Model...")

# Get number of input features
input_dim = X_train_scaled.shape[1]
print(f"Input dimension (number of features): {input_dim}")

# Create MLPClassifier model with similar architecture
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers with 64 and 32 neurons
    activation='relu',
    solver='adam',
    batch_size=32,
    max_iter=50,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.2
)

print("\nModel Architecture:")
print(f"  - Input Layer: {input_dim} features")
print(f"  - Hidden Layer 1: 64 neurons (ReLU activation)")
print(f"  - Hidden Layer 2: 32 neurons (ReLU activation)")
print(f"  - Output Layer: 1 neuron (Logistic/Sigmoid activation)")
print(f"  - Optimizer: Adam")
print(f"  - Max iterations: 50")

# ========================
# 4. TRAIN MODEL
# ========================
print("\n[4] Training Model...")

model.fit(X_train_scaled, y_train)

print("\nTraining completed!")

# ========================
# 5. EVALUATE MODEL
# ========================
print("\n[5] Evaluating Model...")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate accuracy
test_accuracy = model.score(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ========================
# 6. CONFUSION MATRIX
# ========================
print("\n[6] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Calculate additional metrics
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

print("\nDetailed Metrics:")
print(f"  True Positives (TP): {tp}")
print(f"  True Negatives (TN): {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  Sensitivity (Recall): {sensitivity:.4f}")
print(f"  Specificity: {specificity:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  F1-Score: {f1_score:.4f}")

# ========================
# 7. VISUALIZATIONS
# ========================
print("\n[7] Generating Visualizations...")

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Diabetes ANN Model - Evaluation Results', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Plot 2: Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score']
metrics_values = [accuracy, precision, sensitivity, specificity, f1_score]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

bars = axes[1].bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
axes[1].set_title('Performance Metrics', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Score')
axes[1].set_ylim([0, 1])
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/diabetes_ann_results.png', dpi=300, bbox_inches='tight')
print("Visualizations saved as 'diabetes_ann_results.png'")
plt.show()

# ========================
# 8. SAVE MODEL
# ========================
print("\n[8] Saving Model...")

import pickle
with open('../models/diabetes_ann_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as 'diabetes_ann_model.pkl'")

# Save scaler and label encoders
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('../models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("Scaler and encoders saved!")

print("\n" + "=" * 70)
print("MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  - Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Precision: {precision*100:.2f}%")
print(f"  - Sensitivity: {sensitivity*100:.2f}%")
print(f"  - Specificity: {specificity*100:.2f}%")
print(f"  - F1-Score: {f1_score:.4f}")
print("=" * 70)

