# Flexible ML Predictor

## Short Description

A robust, **memory-efficient** command-line tool designed to handle, clean, and analyze **any tabular dataset** (CSV, Excel, JSON). It employs advanced **scikit-learn Pipelines** to prevent data leakage, automatically engineer **datetime features**, select the best model using **Randomized Search**, and provide **feature importance**—making machine learning fast, reliable, and accessible for both regression and classification tasks.

---

## Key Features

* **Data Leakage Prevention:** Utilizes `ColumnTransformer` to ensure preprocessing steps (scaling, imputation) are learned *only* from the training data.
* **Performance & Memory Optimized:** Uses **RandomizedSearchCV** for fast tuning and **sparse encoding/downcasting** for memory efficiency on large datasets.
* **Automatic Feature Engineering:** Automatically detects date columns and extracts predictive features (**month, day, hour**).
* **Dual ML Tasks:** Supports **Regression** and **Classification** with automatic task detection.
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