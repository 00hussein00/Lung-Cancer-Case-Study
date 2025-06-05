# Lung Cancer Survival Prediction

A machine learning project that builds and compares several models to predict lung cancer patient survival based on clinical features. This repository contains data preprocessing scripts, model training notebooks, evaluation metrics, and documentation.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Modeling](#modeling)  
- [Evaluation](#evaluation)  
- [File Structure](#file-structure)  
- [Conclusion & Future Work](#conclusion--future-work)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgments](#acknowledgments)  

---

## Project Overview

Lung cancer remains one of the leading causes of cancer-related deaths worldwide. Accurate survival predictions can guide clinical decisions and personalize treatment plans. This project:

1. **Preprocesses** a public lung cancer dataset (demographics, clinical stage, family history, etc.).  
2. **Trains** and compares multiple classifiers, including K‑Nearest Neighbors, Decision Tree, Random Forest, XGBoost, and a feedforward neural network.  
3. **Evaluates** each model using metrics such as accuracy, precision, recall, and F1‑score.  
4. **Analyzes** feature importance and offers insights into which clinical variables most strongly predict survival.  
5. Provides a **notebook** for reproducibility and future extensions (e.g., survival‑analysis, multimodal data).

---

## Dataset

- **Source**: Publicly available lung cancer dataset (CSV).  
- **Variables**:  
  - `age` (numeric)  
  - `gender` (categorical: Male/Female)  
  - `stage` (categorical: I/II/III/IV)  
  - `family_history` (binary: 0 = no, 1 = yes)  
  - `smoking_history` (e.g., pack‑years)  
  - `survived` (binary target: 0 = did not survive, 1 = survived)  

> **Note:** If you have an updated or extended dataset with additional features (biomarkers, imaging, comorbidities), you can replace the CSV under `data/` and adjust the preprocessing notebook accordingly.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/lung-cancer-survival.git
   cd lung-cancer-survival
   ```

2. **Create a virtual environment** (recommended)  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Linux/macOS
   # venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

   > **Dependencies include**:  
   > - `pandas`  
   > - `numpy`  
   > - `scikit-learn`  
   > - `xgboost`  
   > - `tensorflow` (for Keras)  
   > - `matplotlib` (for visualizations)  
   > - `jupyter` or `jupyterlab` (for running notebooks)  

4. **Verify installation**  
   ```bash
   python -c "import pandas, sklearn, xgboost, tensorflow; print('All imports successful!')"
   ```

---

## Usage

1. **Prepare data**  
   - Place your lung cancer CSV in the `data/` folder and name it `lung_cancer.csv`.  
   - If using a custom filename or path, update the file path in the preprocessing notebook (`notebooks/01_data_preprocessing.ipynb`).

2. **Launch Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

3. **Run Notebooks in Order**  
   1. **01_data_preprocessing.ipynb**  
      - Cleans missing values, encodes categorical features, and splits into train/test sets.  
   2. **02_model_training.ipynb**  
      - Trains KNN, Decision Tree, Random Forest, XGBoost, and a simple Keras neural network.  
      - Saves each trained model (if desired) under `models/`.  
   3. **03_evaluation_and_analysis.ipynb**  
      - Computes accuracy, precision, recall, F1‑score on the test set.  
      - Generates feature‑importance plots for tree‑based models.  
      - (Optional) Runs SHAP analysis if SHAP library is installed.

4. **Configuration**  
   - Modify hyperparameters directly in the notebook cells or create a `config.yaml` if you prefer centralized configuration.  
   - Example:  
     ```yaml
     preprocessing:
       test_size: 0.3
       random_state: 42
     models:
       knn:
         n_neighbors: 5
       random_forest:
         n_estimators: 100
         max_depth: 10
       xgboost:
         learning_rate: 0.1
         max_depth: 6
       neural_network:
         epochs: 50
         batch_size: 32
         learning_rate: 0.001
     ```

---

## Modeling

| Model                  | Description                                                  |
|------------------------|--------------------------------------------------------------|
| **K‑Nearest Neighbors**    | Distance‑based classification; simple baseline.                  |
| **Decision Tree**          | Interpretable tree model; can overfit if not pruned.            |
| **Random Forest**          | Ensemble of decision trees; generally robust to overfitting.   |
| **XGBoost**                | Gradient boosting library; often yields state‑of‑the‑art results. |
| **Neural Network (Keras)** | Feedforward network (dense layers + dropout); requires tuning.  |

All models are implemented via scikit‑learn (except XGBoost’s own API and Keras for the neural network). Check the notebooks for training loops, hyperparameter settings, and model serialization (saved under `models/` as `.pkl` or `.h5` files).

---

## Evaluation

After training, each model is evaluated on the held‑out test set (30 % of data). Key metrics:

- **Accuracy**  
- **Precision (Non‑Survivors)**  
- **Recall (Non‑Survivors)**  
- **F1‑Score (Non‑Survivors)**  

Example results (may vary based on dataset split and hyperparameters):

| Model                 | Test Accuracy | Precision (Non‑Surv) | Recall (Non‑Surv) | F1‑Score (Non‑Surv) |
|-----------------------|---------------|----------------------|-------------------|---------------------|
| K‑Nearest Neighbors   | 0.75          | 0.68                 | 0.72              | 0.70                |
| Decision Tree         | 0.78          | 0.74                 | 0.76              | 0.75                |
| Random Forest         | 0.82          | 0.78                 | 0.82              | 0.80                |
| XGBoost               | 0.84          | 0.81                 | 0.83              | 0.82                |
| Neural Network (Keras)| 0.83          | 0.79                 | 0.83              | 0.81                |

Feature importance (from Random Forest / XGBoost) typically identifies:  
1. `stage`  
2. `age`  
3. `family_history`  
4. `smoking_history`  

Graphs and detailed analysis are in **03_evaluation_and_analysis.ipynb**.

---

## File Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── lung_cancer.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_and_analysis.ipynb
├── models/
│   ├── knn_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── neural_network_model.h5
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_models.py
│   └── evaluate.py
└── LICENSE
```

- **data/** – Contains the raw CSV(s).  
- **notebooks/** – Jupyter notebooks for preprocessing, training, and evaluation.  
- **models/** – Serialized model files (optional; generated after training).  
- **src/** – (Optional) Modular scripts for data loading, preprocessing, training, and evaluation.  
- **requirements.txt** – Python package dependencies.  
- **LICENSE** – Project license (e.g., MIT).

---

## Conclusion & Future Work

See the [Conclusion & Future Work](#conclusion--future-work) section in this README for a summary. For an in‑depth write‑up, refer to the “Conclusion and Future Work” cells in the notebooks.

---

### Conclusion

- Tree‑based ensembles (Random Forest, XGBoost) and the neural network outperformed simpler baselines.  
- Key predictive variables included cancer stage, patient age, and family history.  
- Neural network performance was competitive but required careful tuning (learning rate, dropout, batch size).

### Future Work

1. **Class Imbalance Handling**  
   - Implement SMOTE or class‑weighting to improve minority‑class recall.  

2. **Additional Clinical Features**  
   - Integrate detailed smoking history (pack‑years), genetic biomarkers (EGFR status), and comorbidities (COPD, diabetes).  

3. **Survival‑Analysis Models**  
   - Build a Cox proportional hazards model or DeepSurv to predict time‑to‑event rather than binary survival.  

4. **Multimodal Learning**  
   - Combine tabular data with imaging (CT scans, X‑rays) via CNNs and radiomic feature extraction.  

5. **Model Explainability**  
   - Use SHAP values for granular, patient‑level interpretability.  

6. **External Validation & Deployment**  
   - Validate on an independent cohort; package into a REST API (FastAPI/Flask) or a simple web dashboard.  
