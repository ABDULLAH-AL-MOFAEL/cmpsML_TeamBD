# Water Quality  Indexing – cmpsML_TeamBD

## 📌 Project Overview

This project aims to classify water quality using machine learning algorithms: **ANN**, **SVM**, **Decision Tree**, and **KNN**. It includes three major stages:
1. **Data Preprocessing**
2. **Feature Extraction**
3. **Model Training & Evaluation**

Implemented in Python as part of the **CMPS 470/570 Machine Learning** course.  

## 🧩 Folder Structure
<pre> <code>
'''
cmpsML_TeamBD/
├── CODE/
│   ├── INPUT/
│   │   ├── Raw Data/
│   │   │   └── RawData.xlsx
│   │   ├── TEST/
│   │   │   └── preprocessed_data.xlsx
│   │   └── TRAIN/
│   │       └── preprocessed_data.xlsx
│   ├── OUTPUT/
│   │   ├── feature plots/
│   │   │   └── [5 feature plots as .png]
│   │   ├── MODEL/
│   │   │   └── [4 trained model .pkl files]
│   │   ├── model_results/
│   │   │   └── [4 evaluation plots + 2 Excel files]
│   │   ├── extracted_features.xlsx
│   │   └── extracted_features_with_label.xlsx
│   ├── TeamBD_PA1_Data_Preprocessing.py
│   ├── TeamBD_PA2_Feature_Extraction.py
│   └── TeamBD_PA3_ModelTrainingAndEvaluation.py
├── DOC/
│   ├── TeamBD_PA1_Data preprocessing.pdf
│   ├── TeamBD_PA2_Feature Extraction.pdf
│   └── TeamBD_Proposal.pdf
├── OTHER/
│   └── ReadMe.txt

'''
</code> </pre>

##  Pipeline Overview

1. **Preprocessing**  
   File: `TeamBD_PA1_Data_Preprocessing.py`  
   - Median imputation for missing values
   - Outlier removal using IQR
   - Z-score standardization
   - 60-20-20 train/validation/test split
   - Saves: `preprocessed_data.xlsx`

2. **Feature Extraction**  
   File: `TeamBD_PA2_Feature_Extraction.py`  
   - Features: Mean, Absolute, Log for each parameter
   - Supports interactive file selection (Tkinter GUI)
   - Saves: `extracted_features.xlsx` and plots in `feature plots/`

3. **Model Training & Evaluation**  
   File: `TeamBD_PA3_ModelTrainingAndEvaluation.py`  
   - Trains ANN, SVM, DT, KNN with hyperparameter tuning
   - 3-tier validation + 5-fold cross-validation
   - Evaluation metrics: Accuracy, Precision, Recall, Specificity, F1, AUC
   - Robustness tested over 5 random seeds
   - UI-supported inference mode to load `.pkl` models
   - Saves: `.pkl` models + plots + metrics

