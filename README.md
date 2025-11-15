# Flexible ML Predictor

## Description

An optimized, end-to-end Auto-ML command-line tool for **Supervised** (Regression/Classification) and **Unsupervised** (Clustering/PCA) analysis of tabular data (CSV, Excel, JSON). It guarantees model integrity by employing scikit-learn Pipelines to prevent data leakage and boosts accuracy through automatic feature engineering (datetime extraction). The system prioritizes speed and efficiency by using Randomized Search for fast model selection and memory-optimized processing for large datasets.

---

## Key Features

* **Data Leakage Prevention:** Utilizes `ColumnTransformer` to ensure preprocessing steps (scaling, imputation) are learned *only* from the training data.
* **Performance & Memory Optimized:** Uses **RandomizedSearchCV** for fast tuning and **sparse encoding/downcasting** for memory efficiency on large datasets.
* **Automatic Feature Engineering:** Automatically detects date columns and extracts predictive features (**month, day, hour**).
* **Dual ML Tasks:** Supports **Regression** and **Classification** with automatic task detection.
* **Unsupervised Analysis (NEW):** Provides in-tool options for **Clustering (K-Means)** and **Dimensionality Reduction (PCA)**, adding the resulting analysis columns directly to the DataFrame.
* **Feature Importance:** Reports the **Top 10 most influential features** after training, boosting interpretability.
* **Model Persistence:** Save and load the entire trained pipeline (preprocessor + model) using `joblib` for re-use.
* **Deep Analysis:** Efficient **Pandas-based querying** and data visualization (histograms, scatter plots).

---

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/flexible-ml-predictor.git
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

    ```bash
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Run the main application from your terminal to access the interactive menu:

```bash
python main.py
```