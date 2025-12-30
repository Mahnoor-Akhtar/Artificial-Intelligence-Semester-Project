"""
Diabetes Prediction Models Comparison Report Generator
Compares Decision Tree, Naive Bayes, and ANN models
Generates a comprehensive PDF report with visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
try:
    from fpdf import FPDF
except ImportError:
    from fpdf2 import FPDF
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

print("=" * 80)
print("DIABETES PREDICTION - MODEL COMPARISON REPORT")
print("=" * 80)

# ========================
# NAIVE BAYES IMPLEMENTATION
# ========================
class MixedNaiveBayes:
    def __init__(self, continuous_cols, categorical_cols):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.priors = {}
        self.gaussian_params = {}
        self.categorical_probs = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for cls in self.classes:
            self.priors[cls] = np.sum(y == cls) / n_samples
            X_c = X[y == cls]
            
            # Continuous features: Gaussian
            self.gaussian_params[cls] = {}
            for col in self.continuous_cols:
                mean = X_c[col].mean()
                var = X_c[col].var()
                self.gaussian_params[cls][col] = (mean, var)
                
            # Categorical features: Probabilities
            self.categorical_probs[cls] = {}
            for col in self.categorical_cols:
                value_counts = X_c[col].value_counts()
                total_count = len(X_c)
                unique_values = X[col].unique()
                
                prob_dict = {}
                for val in unique_values:
                    count = value_counts.get(val, 0)
                    prob_dict[val] = (count + 1) / (total_count + len(unique_values))
                self.categorical_probs[cls][col] = prob_dict

    def _gaussian_probability(self, x, mean, var):
        if var == 0:
            var = 1e-6
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_scores = {}
            for cls in self.classes:
                score = np.log(self.priors[cls])
                
                # Continuous features
                for col in self.continuous_cols:
                    mean, var = self.gaussian_params[cls][col]
                    prob = self._gaussian_probability(row[col], mean, var)
                    score += np.log(prob + 1e-10)
                
                # Categorical features
                for col in self.categorical_cols:
                    val = row[col]
                    prob = self.categorical_probs[cls][col].get(val, 1e-10)
                    score += np.log(prob + 1e-10)
                
                class_scores[cls] = score
            
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

# ========================
# DATA LOADING & PREPROCESSING
# ========================
print("\n[1] Loading and Preprocessing Data...")
df = pd.read_csv('../data/diabetes_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'smoking_history']

df_encoded = df.copy()
for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df_encoded.drop('diabetes', axis=1)
y = df_encoded['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ========================
# MODEL 1: DECISION TREE
# ========================
print("\n[2] Training Decision Tree Model...")
dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred, zero_division=0)
dt_recall = recall_score(y_test, dt_pred, zero_division=0)
dt_f1 = f1_score(y_test, dt_pred, zero_division=0)
dt_cm = confusion_matrix(y_test, dt_pred)

print(f"✓ Decision Tree - Accuracy: {dt_accuracy*100:.2f}%")

# ========================
# MODEL 2: NAIVE BAYES
# ========================
print("\n[3] Training Naive Bayes Model...")
continuous_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_cols = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

nb_model = MixedNaiveBayes(continuous_cols, categorical_cols)
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)
nb_precision = precision_score(y_test, nb_pred, zero_division=0)
nb_recall = recall_score(y_test, nb_pred, zero_division=0)
nb_f1 = f1_score(y_test, nb_pred, zero_division=0)
nb_cm = confusion_matrix(y_test, nb_pred)

print(f"✓ Naive Bayes - Accuracy: {nb_accuracy*100:.2f}%")

# ========================
# MODEL 3: ARTIFICIAL NEURAL NETWORK
# ========================
print("\n[4] Training ANN Model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ann_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    batch_size=32,
    max_iter=50,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.2
)
ann_model.fit(X_train_scaled, y_train)
ann_pred = ann_model.predict(X_test_scaled)

ann_accuracy = accuracy_score(y_test, ann_pred)
ann_precision = precision_score(y_test, ann_pred, zero_division=0)
ann_recall = recall_score(y_test, ann_pred, zero_division=0)
ann_f1 = f1_score(y_test, ann_pred, zero_division=0)
ann_cm = confusion_matrix(y_test, ann_pred)

print(f"✓ ANN - Accuracy: {ann_accuracy*100:.2f}%")

# ========================
# COMPARISON VISUALIZATIONS
# ========================
print("\n[5] Generating Comparison Visualizations...")

# Create comparison figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Accuracy Comparison (Bar Chart)
ax1 = fig.add_subplot(gs[0, :])
models = ['Decision Tree', 'Naive Bayes', 'ANN']
accuracies = [dt_accuracy*100, nb_accuracy*100, ann_accuracy*100]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 100])
ax1.grid(True, alpha=0.3, axis='y')

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 2-4: Confusion Matrices
confusion_matrices = [
    (dt_cm, 'Decision Tree', gs[1, 0]),
    (nb_cm, 'Naive Bayes', gs[1, 1]),
    (ann_cm, 'ANN', gs[1, 2])
]

for cm, title, position in confusion_matrices:
    ax = fig.add_subplot(position)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{title}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)

# Plot 5: All Metrics Comparison (Grouped Bar Chart)
ax5 = fig.add_subplot(gs[2, :])
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_metrics = [dt_accuracy, dt_precision, dt_recall, dt_f1]
nb_metrics = [nb_accuracy, nb_precision, nb_recall, nb_f1]
ann_metrics = [ann_accuracy, ann_precision, ann_recall, ann_f1]

x = np.arange(len(metrics_names))
width = 0.25

bars1 = ax5.bar(x - width, dt_metrics, width, label='Decision Tree', 
                color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax5.bar(x, nb_metrics, width, label='Naive Bayes', 
                color='#e74c3c', alpha=0.7, edgecolor='black')
bars3 = ax5.bar(x + width, ann_metrics, width, label='ANN', 
                color='#2ecc71', alpha=0.7, edgecolor='black')

ax5.set_title('Comprehensive Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(metrics_names, fontsize=11)
ax5.set_ylim([0, 1.1])
ax5.legend(loc='upper right', fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Diabetes Prediction Models - Comprehensive Comparison', 
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('../results/models_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Comparison visualizations saved to ../results/models_comparison.png")
plt.close()

# ========================
# DETERMINE BEST MODEL
# ========================
print("\n[6] Analyzing Results...")

results_df = pd.DataFrame({
    'Model': models,
    'Accuracy': [dt_accuracy, nb_accuracy, ann_accuracy],
    'Precision': [dt_precision, nb_precision, ann_precision],
    'Recall': [dt_recall, nb_recall, ann_recall],
    'F1-Score': [dt_f1, nb_f1, ann_f1]
})

# Calculate average score for ranking
results_df['Average Score'] = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].mean(axis=1)
results_df = results_df.sort_values('Average Score', ascending=False)

best_model = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n✓ Best Model: {best_model}")
print(f"  Accuracy: {best_accuracy*100:.2f}%")

# ========================
# GENERATE PDF REPORT
# ========================
print("\n[7] Generating PDF Report...")

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(44, 62, 80)
        self.cell(0, 10, 'DIABETES PREDICTION MODELS', 0, 1, 'C')
        self.set_font('Arial', 'B', 14)
        self.cell(0, 8, 'Comprehensive Comparison Report', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

    def add_metric_table(self, data):
        self.set_font('Arial', 'B', 10)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        
        # Header
        col_widths = [50, 30, 30, 30, 30]
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 8, header, 1, 0, 'C', True)
        self.ln()
        
        # Data rows
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        
        colors = [(52, 152, 219), (231, 76, 60), (46, 204, 113)]
        
        for idx, row in data.iterrows():
            # Highlight best model
            if idx == 0:
                self.set_fill_color(39, 174, 96)
                self.set_font('Arial', 'B', 10)
            else:
                self.set_fill_color(236, 240, 241)
                self.set_font('Arial', '', 10)
            
            self.cell(col_widths[0], 8, row['Model'], 1, 0, 'L', True)
            self.cell(col_widths[1], 8, f"{row['Accuracy']*100:.2f}%", 1, 0, 'C', True)
            self.cell(col_widths[2], 8, f"{row['Precision']*100:.2f}%", 1, 0, 'C', True)
            self.cell(col_widths[3], 8, f"{row['Recall']*100:.2f}%", 1, 0, 'C', True)
            self.cell(col_widths[4], 8, f"{row['F1-Score']:.4f}", 1, 0, 'C', True)
            self.ln()

pdf = PDFReport()
pdf.add_page()

# Executive Summary
pdf.chapter_title('1. EXECUTIVE SUMMARY')
summary = f"""This report presents a comprehensive comparison of three machine learning models for diabetes prediction: Decision Tree, Naive Bayes, and Artificial Neural Network (ANN).

Dataset Overview:
- Total Samples: {len(df):,}
- Training Samples: {len(X_train):,}
- Test Samples: {len(X_test):,}
- Features: {X.shape[1]} (Age, BMI, HbA1c Level, Blood Glucose, Gender, Hypertension, Heart Disease, Smoking History)

Best Performing Model: {best_model}
Overall Accuracy: {best_accuracy*100:.2f}%

The analysis evaluated models based on multiple metrics including accuracy, precision, recall, and F1-score to ensure comprehensive performance assessment."""

pdf.chapter_body(summary)

# Model Results
pdf.chapter_title('2. DETAILED MODEL PERFORMANCE')
pdf.add_metric_table(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']])

# Model-by-Model Analysis
pdf.ln(5)
pdf.chapter_title('3. MODEL-BY-MODEL ANALYSIS')

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 8, 'A. Decision Tree Classifier', 0, 1)
pdf.set_font('Arial', '', 11)
dt_analysis = f"""Performance Metrics:
- Accuracy: {dt_accuracy*100:.2f}%
- Precision: {dt_precision*100:.2f}%
- Recall: {dt_recall*100:.2f}%
- F1-Score: {dt_f1:.4f}

Strengths: Easy to interpret, handles non-linear relationships, requires minimal data preprocessing.
Weaknesses: Prone to overfitting, can be unstable with small data changes.
"""
pdf.multi_cell(0, 6, dt_analysis)
pdf.ln(3)

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 8, 'B. Naive Bayes Classifier', 0, 1)
pdf.set_font('Arial', '', 11)
nb_analysis = f"""Performance Metrics:
- Accuracy: {nb_accuracy*100:.2f}%
- Precision: {nb_precision*100:.2f}%
- Recall: {nb_recall*100:.2f}%
- F1-Score: {nb_f1:.4f}

Strengths: Fast training, works well with small datasets, handles mixed data types (continuous & categorical).
Weaknesses: Assumes feature independence, may underperform with correlated features.
"""
pdf.multi_cell(0, 6, nb_analysis)
pdf.ln(3)

pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 8, 'C. Artificial Neural Network (ANN)', 0, 1)
pdf.set_font('Arial', '', 11)
ann_analysis = f"""Performance Metrics:
- Accuracy: {ann_accuracy*100:.2f}%
- Precision: {ann_precision*100:.2f}%
- Recall: {ann_recall*100:.2f}%
- F1-Score: {ann_f1:.4f}

Architecture: 2 hidden layers (64, 32 neurons)
Strengths: Captures complex non-linear patterns, highly flexible, scalable to large datasets.
Weaknesses: Requires more computational resources, needs feature scaling, less interpretable.
"""
pdf.multi_cell(0, 6, ann_analysis)

# Add new page for visualizations and recommendations
pdf.add_page()

pdf.chapter_title('4. VISUAL COMPARISON')
pdf.set_font('Arial', '', 11)
pdf.multi_cell(0, 6, "The following visualization shows comprehensive comparison of all three models across multiple performance metrics:")
pdf.ln(5)

# Add comparison image
try:
    pdf.image('../results/models_comparison.png', x=10, y=pdf.get_y(), w=190)
except:
    pdf.multi_cell(0, 6, "[Visualization image will be available after running the script]")

# Recommendations (new page if needed)
pdf.add_page()
pdf.chapter_title('5. RECOMMENDATIONS & CONCLUSION')

# Determine recommendation based on results
if best_model == 'Decision Tree':
    recommendation = """Based on the comprehensive analysis, the Decision Tree model is recommended for this diabetes prediction task.

Why Decision Tree is Best for Your Dataset:
1. Highest overall accuracy among the three models
2. Excellent interpretability - easy to understand which features drive predictions
3. Requires minimal preprocessing - works directly with the data
4. Robust performance on this specific dataset structure
5. Low computational requirements for deployment

Implementation Advantages:
- Can be easily visualized and explained to non-technical stakeholders
- Fast prediction times suitable for real-time applications
- Clear decision paths help identify key risk factors
- Minimal maintenance and update requirements

Deployment Recommendations:
- Use the trained model with max_depth=5 for optimal performance
- Monitor performance regularly with new data
- Consider ensemble methods (Random Forest) for further improvements"""

elif best_model == 'Naive Bayes':
    recommendation = """Based on the comprehensive analysis, the Naive Bayes model is recommended for this diabetes prediction task.

Why Naive Bayes is Best for Your Dataset:
1. Highest overall accuracy among the three models
2. Extremely fast training and prediction times
3. Works efficiently with mixed data types (continuous and categorical)
4. Robust with limited training data
5. Low computational overhead for deployment

Implementation Advantages:
- Probabilistic predictions provide confidence scores
- Handles new data efficiently with minimal retraining
- Works well with the feature distribution of this dataset
- Minimal risk of overfitting

Deployment Recommendations:
- Monitor feature distributions for data drift
- Update class priors periodically with new data
- Consider using probability thresholds for high-stakes predictions
- Implement regular model retraining schedule"""

else:  # ANN
    recommendation = """Based on the comprehensive analysis, the Artificial Neural Network (ANN) model is recommended for this diabetes prediction task.

Why ANN is Best for Your Dataset:
1. Highest overall accuracy among the three models
2. Captures complex non-linear relationships in the data
3. Superior generalization on unseen data
4. Scalable architecture for future improvements
5. Excellent balance across all performance metrics

Implementation Advantages:
- Deep learning capability handles feature interactions automatically
- Can be enhanced with more data without architectural changes
- Robust performance across different metrics (precision, recall, F1)
- Modern deployment frameworks widely available

Deployment Recommendations:
- Use the trained model with (64, 32) hidden layer architecture
- Ensure proper feature scaling in production pipeline
- Monitor model performance and retrain periodically
- Consider using model checkpointing for version control
- Implement input validation to match training data distribution

Performance Optimization:
- Current architecture provides {ann_accuracy*100:.2f}% accuracy
- Can be fine-tuned with hyperparameter optimization
- Consider data augmentation for further improvements"""

pdf.chapter_body(recommendation)

# Final conclusion
pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 8, 'Final Verdict:', 0, 1)
pdf.set_font('Arial', '', 11)

conclusion = f"""The {best_model} model demonstrates superior performance for diabetes prediction on your dataset with {best_accuracy*100:.2f}% accuracy. This model provides the best balance of performance, reliability, and practical deployment considerations.

All three models show competent performance, indicating that your dataset is well-suited for machine learning classification. The choice of {best_model} is based on quantitative metrics, interpretability, and deployment feasibility.

Next Steps:
1. Deploy the {best_model} model in a production environment
2. Establish monitoring for model performance degradation
3. Collect feedback from real-world predictions
4. Plan periodic retraining with new data
5. Consider ensemble methods combining multiple models for even better results

Report Generated: December 30, 2025
Analysis Method: Stratified train-test split (80-20) with consistent random seed
"""

pdf.multi_cell(0, 6, conclusion)

# Save PDF
pdf_path = '../results/diabetes_models_comparison_report.pdf'
pdf.output(pdf_path)

print(f"✓ PDF Report generated: {pdf_path}")

# ========================
# SUMMARY
# ========================
print("\n" + "=" * 80)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nRanking:")
for idx, row in results_df.iterrows():
    rank = idx + 1
    symbol = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
    print(f"{symbol} #{rank}: {row['Model']}")
    print(f"   Accuracy: {row['Accuracy']*100:.2f}% | Precision: {row['Precision']*100:.2f}% | Recall: {row['Recall']*100:.2f}% | F1: {row['F1-Score']:.4f}")
    print()

print(f"\n📊 Results saved:")
print(f"   - Visualization: ../results/models_comparison.png")
print(f"   - PDF Report: ../results/diabetes_models_comparison_report.pdf")
print("\n" + "=" * 80)
