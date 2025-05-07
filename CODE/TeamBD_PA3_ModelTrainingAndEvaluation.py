#%% MODULE BEGINS
module_name = 'Model Training and Evaluation'

'''
Version: v4.0

Description:
    Full training and evaluation of ANN, SVM, DT, and KNN models.
    Includes preprocessing, model building, hyperparameter tuning with Validation set,
    5-fold cross-validation, evaluation (Accuracy, Precision, Recall, Specificity, F1, AUC),
    plotting ROC curves, bar plots, robustness boxplots, Epoch vs Error curve for ANN,
    and saving models and metrics.

Authors:
    Abdullah Al Mofael, Fahim Muntasir Rabbi (Updated)

Date Created     : 2025-04-27
Date Last Updated: 2025-04-28

Doc:
    Input: ./OUTPUT/extracted_features_with_label.xlsx
    Output: ./OUTPUT/model_results/performance_metrics.xlsx, plots
    Models: ./MODEL/*.pkl

'''

#%% IMPORTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Standard and ML libraries
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn modules for modeling and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize

#%% PATHS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define key directories and ensure output/model folders exist
CODE_DIR = Path(__file__).resolve().parent
CODE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CODE_DIR / "INPUT"
OUTPUT_DIR = CODE_DIR / "OUTPUT"         
MODEL_DIR = OUTPUT_DIR / "MODEL"
OUTPUT_RESULTS_DIR = OUTPUT_DIR / "model_results"
INPUT_FEATURES_PATH = OUTPUT_DIR / "extracted_features_with_label.xlsx"


MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#%% FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Load pre-split feature-labeled data from Excel

def load_data():
    data = pd.read_excel(INPUT_FEATURES_PATH, sheet_name=None)
    return data['Training'], data['Validation'], data['Testing']


# Separate features and target label

def split_features_labels(df):
    X = df.drop(columns=['Label'])
    y = df['Label']
    return X, y


# Build base ML models

def build_models():
    return {
        'ANN': MLPClassifier(max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.2),
        'SVM': SVC(probability=True, random_state=42),
        'DT': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier()
    }


# Hyperparameter tuning using validation data (only ANN tuned here)

def tune_hyperparameters(model_name, model, X_train, y_train, X_val, y_val):
    if model_name == 'ANN':
        best_model = None
        best_score = -np.inf
        for hls in [(50,), (100,), (50, 50)]:
            for act in ['relu', 'tanh']:
                temp_model = MLPClassifier(hidden_layer_sizes=hls, activation=act, max_iter=500, random_state=42, early_stopping=True)
                temp_model.fit(X_train, y_train)
                score = temp_model.score(X_val, y_val)
                if score > best_score:
                    best_score = score
                    best_model = temp_model
        return best_model
    else:
        model.fit(X_train, y_train)
        return model


# Evaluate trained model using multiple metrics

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    if y_prob is not None:
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        auc_score = np.nan

    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) != 0 else 0

    return acc, prec, rec, specificity, f1, auc_score


# Save model as .pkl file

def save_model(model, name):
    joblib.dump(model, MODEL_DIR / f"{name.lower()}_model.pkl")


# Save evaluation metrics to Excel

def save_metrics(metrics_dict):
    pd.DataFrame(metrics_dict).to_excel(OUTPUT_RESULTS_DIR / "performance_metrics.xlsx", index=False)


# Plot ROC curves using one-vs-rest approach for multi-class

def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(10,8))
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} - Class {class_label} (AUC={roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curves (One-vs-Rest for each class)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(OUTPUT_RESULTS_DIR / "roc_curves.png")
    plt.close()


# Compare Accuracy, F1, AUC for all models using bar plot

def plot_comparison_bars(metrics_df):
    metrics_df.set_index("Model")[['Accuracy', 'F1', 'AUC']].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(OUTPUT_RESULTS_DIR / "performance_comparison.png")
    plt.close()


# Evaluate robustness across random seeds using accuracy boxplot

def robustness_testing(models, X, y):
    scores = {name: [] for name in models.keys()}
    for seed in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        for name, model in models.items():
            temp_model = build_models()[name]
            temp_model.fit(X_train, y_train)
            acc, *_ = evaluate_model(temp_model, X_test, y_test)
            scores[name].append(acc)

    score_df = pd.DataFrame(scores)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=score_df)
    plt.title('Model Robustness (Accuracy over 5 random splits)')
    plt.savefig(OUTPUT_RESULTS_DIR / "robustness_boxplot.png")
    plt.close()


# Perform K-Fold CV and export accuracy results

def k_fold_cross_validation(models, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        temp_model = build_models()[name]
        scores = cross_val_score(temp_model, X, y, cv=skf, scoring='accuracy')
        results[name] = scores
    pd.DataFrame(results).to_excel(OUTPUT_RESULTS_DIR / "kfold_cross_validation_scores.xlsx", index=False)


# Plot training error across epochs for ANN

def plot_ann_epoch_error(model):
    if hasattr(model, 'loss_curve_'):
        plt.figure()
        plt.plot(model.loss_curve_, label='Training Loss')
        plt.title('Epoch vs Training Error Curve (ANN)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_RESULTS_DIR / "ann_epoch_error_curve.png")
        plt.close()

#%% MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main workflow for training, evaluating, saving, and plotting

def main():
    train_data, val_data, test_data = load_data()
    X_train, y_train = split_features_labels(train_data)
    X_val, y_val = split_features_labels(val_data)
    X_test, y_test = split_features_labels(test_data)

    if y_train.dtype == 'O':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

    models = build_models()
    metrics = {"Model": [], "Accuracy": [], "Precision": [], "Recall": [], "Specificity": [], "F1": [], "AUC": []}
    best_models = {}

    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        tuned_model = tune_hyperparameters(name, model, X_train, y_train, X_val, y_val)
        best_models[name] = tuned_model

        acc, prec, rec, spec, f1, auc_score = evaluate_model(tuned_model, X_test, y_test)
        metrics["Model"].append(name)
        metrics["Accuracy"].append(acc)
        metrics["Precision"].append(prec)
        metrics["Recall"].append(rec)
        metrics["Specificity"].append(spec)
        metrics["F1"].append(f1)
        metrics["AUC"].append(auc_score)

        save_model(tuned_model, name)

        if name == 'ANN':
            plot_ann_epoch_error(tuned_model)

    metrics_df = pd.DataFrame(metrics)
    save_metrics(metrics)

    plot_roc_curves(best_models, X_test, y_test)
    plot_comparison_bars(metrics_df)

    full_X = pd.concat([X_train, X_val, X_test], axis=0)
    full_y = np.concatenate([y_train, y_val, y_test], axis=0)

    robustness_testing(models, full_X, full_y)
    k_fold_cross_validation(models, full_X, full_y)

    print("[STATUS] Model Training and Evaluation Completed Successfully!")


#%% LOAD AND RUN SAVED MODEL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Function to load a saved model and run inference on new test data

def run_saved_model_inference():
    """Load saved model and test data, then predict and print results."""  #

    import tkinter as tk
    from tkinter import filedialog
    print("[INFO] Load a model (.pkl) and test data (.xlsx)...")  #

    # GUI dialog to select model file
    root = tk.Tk()
    root.withdraw()
    model_path = filedialog.askopenfilename(title="Select Saved Model File (.pkl)", filetypes=[("Pickle files", "*.pkl")])
    if not model_path:
        print("[ERROR] No model selected.")  #
        return

    # GUI dialog to select test Excel file
    test_path = filedialog.askopenfilename(title="Select Test Excel File", filetypes=[("Excel files", "*.xlsx")])
    if not test_path:
        print("[ERROR] No test file selected.")  #
        return

    try:
        # Load model
        model = joblib.load(model_path)

        # Load test data and prepare
        data = pd.read_excel(test_path, sheet_name='Testing')
        X_test = data.drop(columns=['Label'])
        y_test = data['Label']

        if y_test.dtype == 'O':
            le = LabelEncoder()
            y_test = le.fit_transform(y_test)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[RESULT] Accuracy on selected test data: {acc:.2f}")  #

        # Save predictions to Excel
        pred_df = data.copy()
        pred_df['Predicted'] = y_pred
        output_pred_path = OUTPUT_RESULTS_DIR / "model_inference_predictions.xlsx"
        pred_df.to_excel(output_pred_path, index=False)
        print(f"[INFO] Predictions saved to: {output_pred_path}")  #

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")  #



#%% SELF RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% SELF RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print(f"\"{module_name}\" module begins.")  #
    mode = input("Enter 'train' to train models or 'inference' to run saved model: ").strip().lower()
    if mode == 'train':
        main()
    elif mode == 'inference':
        run_saved_model_inference()
    else:
        print("[ERROR] Invalid option. Please enter 'train' or 'inference'.")  #
