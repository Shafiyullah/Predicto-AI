import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings
import re
import joblib
import gc
from pathlib import Path
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from analyzer.preprocessing import create_preprocessor

warnings.filterwarnings('ignore')

# RL Imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from analyzer.rl_env import DataFrameEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class DataAnalyzerError(Exception):
    """Custom exception for DataAnalyzer errors."""
    pass

class DataAnalyzer:
    """
    Core engine for the Flexible ML Predictor.
    
    Handles data loading, cleaning, preprocessing, model training (Supervised/Unsupervised),
    and Reinforcement Learning tasks. Uses a pipeline approach for robust ML workflows.
    """
    def __init__(self):
        """Initialize the DataAnalyzer with empty state."""
        self.df = None
        self.model_pipeline = None
        self.target_col = None
        self.features_list = None
        self.model_type = None
        self.unsupervised_model = None
        self.unsupervised_type = None
        self.rl_model_path = None
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_file_path(self, file_path: str) -> bool:
        """
        Validates if the provided file path exists and is a supported format.
        Args: file_path (str): Absolute or relative path to the file.
        Returns: bool: True if valid.
        """
        try:
            if not file_path or not isinstance(file_path, str):
                raise ValueError("File path must be a non-empty string")
            path = Path(file_path)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File does not exist: {file_path}")
            allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
            if path.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported format: {path.suffix}. Allowed: {allowed_extensions}")
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024 * 1024: # 100MB
                self.logger.warning(f"File size is large ({file_size / 1024**2:.2f} MB). Processing may be slow.")
            return True
        except Exception as e:
            self.logger.error(f"File validation failed: {str(e)}")
            raise DataAnalyzerError(f"Validation failed: {e}") from e
    
    def load_new_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Loads a dataframe from the specified file path using optimal engines.
        Args: file_path (str): Path to the source file.
        Returns: pd.DataFrame: The loaded raw dataframe.
        """
        self.validate_file_path(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            # Added low_memory=True, OPTIMIZATION: Use PyArrow engine
            df = pd.read_csv(file_path, on_bad_lines='warn', encoding='utf-8', low_memory=True, engine='pyarrow')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=0)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        
        if df.empty:
            raise ValueError("File is empty or contains no readable data")
            
        return df

    def _prepare_data_for_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        # 1. Sanitize column names
        df.columns = [re.sub(r'[^\w\s-]', '', str(col)) for col in df.columns]
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # 2. Aggressively convert to numeric where possible
        # Pre-calculate len for efficiency
        n_rows = len(df)
        threshold = 0.8 * n_rows
        
        for col in df.columns:
            # Skip if already numeric or datetime
            if is_numeric_dtype(df[col]) or is_datetime64_any_dtype(df[col]):
                continue
                
            # Attempt numeric conversion
            converted_numeric = pd.to_numeric(df[col], errors='coerce')
            if converted_numeric.count() > threshold:
                df[col] = converted_numeric
                self.logger.info(f"Converted column '{col}' to numeric.")
            
            try:
                converted_datetime = pd.to_datetime(df[col], errors='coerce')
                # If >80% are valid dates, convert the column
                if converted_datetime.count() > threshold:
                    df[col] = converted_datetime
                    self.logger.info(f"Converted column '{col}' to datetime.")
            except Exception:
                pass

        # 3. Sanitize string columns
        # Vectorized string substitution for entire subset if possible
        string_cols = df.select_dtypes(include=['object']).columns
        if not string_cols.empty:
            # Using apply with optimized str accessor is usually fastest for string ops in pandas < 3.0
            # but straightforward assignment is safer.
            for col in string_cols:
                # Ensure string type then strip and clean
                s = df[col].astype(str).str.strip()
                df[col] = s.replace(r'^[<>"\';\n\r]+$', '', regex=True)
        
        return df
    
    def _downcast_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ MEMORY : Downcast numeric columns to save memory."""
        # Use select_dtypes for cleaner selection
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns
        
        # Downcast float types
        for col in fcols:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        # Downcast integer types
        for col in icols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        self.logger.info(f"Downcasted {len(fcols)} float and {len(icols)} integer columns.")
        return df

    def load_data(self, file_path: str):
        """
        Orchestrates the data loading pipeline: Load -> Prepare -> Downcast -> Deduplicate.
        
        Args:
            file_path (str): Path to the dataset.
        """
        try:
            df = self.load_new_dataframe(file_path)
            df = self._prepare_data_for_processing(df)
            df = self._downcast_numerics(df)
            # Drop duplicates
            original_shape = df.shape
            df = df.drop_duplicates()
            cleaned_shape = df.shape
            self.logger.info(f"Dropped {original_shape[0] - cleaned_shape[0]} duplicate rows.")
            
            self.df = df
            self.logger.info(f"Data loaded and prepared: {len(df)} rows, {len(df.columns)} columns.")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise DataAnalyzerError(f"Failed to load data: {e}") from e

    def get_column_types(self) -> dict:
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        # Datetime cols
        datetime_features = self.df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        
        # Numeric cols
        numeric_features = self.df.select_dtypes(include=np.number).columns.tolist()
        
        # Categorical cols
        categorical_features = []
        # Exclude numeric and datetime columns we've already found
        potential_cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        for col in potential_cat_cols:
            nunique = self.df[col].nunique()
            if nunique <= 100:
                categorical_features.append(col)
            else:
                self.logger.warning(f"Dropping high-cardinality (>{nunique}) string column: '{col}'")
                
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'datetime': datetime_features
        }

    def train_model(self, target_col: str):
        """
        Trains multiple models (Reg/Class) and selects the best one via Cross-Validation.
        Args: target_col (str): The name of the target column to predict.
        Raises: DataAnalyzerError: If training fails.
        """
        if self.df is None: raise DataAnalyzerError("Data not loaded.")
        if target_col not in self.df.columns: raise DataAnalyzerError(f"Target column '{target_col}' not found.")
            
        self.target_col = target_col
        
        # Identify Features (X) and Target (y)
        column_types = self.get_column_types()
        
        if target_col in column_types['numeric']:
            self.model_type = 'regression'
            y = self.df[target_col]
        elif target_col in column_types['categorical']:
            self.model_type = 'classification'
            y, _ = pd.factorize(self.df[target_col])
        else:
            raise ValueError(f"Target column '{target_col}' is not a valid numeric or categorical type.")

        self.logger.info(f"Detected task type: {self.model_type}")

        # Features are all valid columns *except* the target
        numeric_features = [col for col in column_types['numeric'] if col != target_col]
        categorical_features = [col for col in column_types['categorical'] if col != target_col]
        datetime_features = [col for col in column_types['datetime'] if col != target_col]
        self.features_list = numeric_features + categorical_features + datetime_features
        
        X = self.df[self.features_list]
        if X.empty: raise ValueError("No feature columns available. Check data.")
            
        # Split Data FIRST
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create Preprocessing Pipeline
        preprocessor = create_preprocessor(numeric_features, categorical_features, datetime_features)
        
        # Define Models and Hyperparameter Grids 
        if self.model_type == 'regression':
            best_pipeline, best_model_name, best_score = self._train_regression(X_train, y_train, X_test, y_test, preprocessor)
        else:
            best_pipeline, best_model_name, best_score = self._train_classification(X_train, y_train, X_test, y_test, preprocessor)
            
        self.model_pipeline = best_pipeline
        self.logger.info(f"Selected best model: {best_model_name} with CV score: {best_score:.4f}")
        
        # Final Evaluation & Feature Importance
        y_pred = self.model_pipeline.predict(X_test)
        
        print("\n--- Model Training Complete ---")
        print(f"**Best Model:** {best_model_name}")
        print(f"**Features Used:** {self.features_list}")
        
        if self.model_type == 'regression':
            print(f"**Test Set R² Score:** {r2_score(y_test, y_pred):.4f}")
            print(f"**Test Set MSE:** {mean_squared_error(y_test, y_pred):.4f}")
        else:
            print(f"**Test Set Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            print("\n**Test Set Classification Report:**")
            print(classification_report(y_test, y_pred, zero_division=0))
        
        # Feature Importance 
        self.show_feature_importance()
        
        # Force garbage collection
        gc.collect()

    def _train_regression(self, X_train, y_train, X_test, y_test, preprocessor):
        models = {
            'LinearRegression': (LinearRegression(), {}),
            'RandomForest': (RandomForestRegressor(random_state=42), 
                             {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, 20, None]}),
            'GradientBoosting': (GradientBoostingRegressor(random_state=42), 
                                 {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]}),
            'HistGradientBoosting': (HistGradientBoostingRegressor(random_state=42),
                                     {'model__learning_rate': [0.05, 0.1], 'model__max_iter': [50, 100]})
        }
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = (XGBRegressor(random_state=42, enable_categorical=True),
                                 {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.05, 0.1, 0.2]})
        return self._train_models_generic(models, X_train, y_train, X_test, y_test, preprocessor, 'r2')

    def _train_classification(self, X_train, y_train, X_test, y_test, preprocessor):
        models = {
            'LogisticRegression': (LogisticRegression(random_state=42, max_iter=1000), 
                                   {'model__C': [0.1, 1.0, 10]}),
            'RandomForest': (RandomForestClassifier(random_state=42), 
                             {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, 20, None]}),
            'GradientBoosting': (GradientBoostingClassifier(random_state=42), 
                                 {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]}),
            'HistGradientBoosting': (HistGradientBoostingClassifier(random_state=42),
                                     {'model__learning_rate': [0.05, 0.1], 'model__max_iter': [50, 100]})
        }
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = (XGBClassifier(random_state=42, enable_categorical=True),
                                 {'model__n_estimators': [50, 100, 200], 'model__learning_rate': [0.05, 0.1, 0.2]})
        return self._train_models_generic(models, X_train, y_train, X_test, y_test, preprocessor, 'accuracy')

    def _train_models_generic(self, models, X_train, y_train, X_test, y_test, preprocessor, scoring):
        best_score = -np.inf
        best_pipeline = None
        best_model_name = ""

        for name, (model, params) in models.items():
            self.logger.info(f"--- Training {name} ---")
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            if not params:
                try:
                    pipeline.fit(X_train, y_train)
                    score = pipeline.score(X_test, y_test)
                    search = pipeline
                except Exception as e:
                    self.logger.warning(f"Failed to train {name}. Error: {e}")
                    continue
            else:
                try:
                    search = RandomizedSearchCV(pipeline, params, n_iter=10, cv=3, scoring=scoring, n_jobs=-1, random_state=42)
                    search.fit(X_train, y_train)
                    score = search.best_score_
                except Exception as e:
                    self.logger.warning(f"Failed to train {name}. Error: {e}")
                    continue

            self.logger.info(f"Best CV score for {name} ({scoring}): {score:.4f}")
            if score > best_score:
                best_score = score
                best_pipeline = search.best_estimator_ if hasattr(search, 'best_estimator_') else search
                best_model_name = name

        if best_pipeline is None: raise RuntimeError("All models failed to train.")
        return best_pipeline, best_model_name, best_score

    def show_feature_importance(self):
        """
        Extracts and displays feature importance from the trained pipeline.
        Prints the top 10 features to stdout.
        """
        if self.model_pipeline is None:
            print("No model is trained.")
            return

        try:
            preprocessor = self.model_pipeline.named_steps['preprocessor']
            model = self.model_pipeline.named_steps['model']
            
            feature_names = preprocessor.get_feature_names_out()
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                importance_type = "Importance"

            elif hasattr(model, 'coef_'):
                if model.coef_.ndim > 1:
                    importances = np.mean(np.abs(model.coef_), axis=0)
                else:
                    importances = np.abs(model.coef_)
                importance_type = "Coefficient (Abs)"

            else:
                print("\nFeature importance not available for this model type.")
                return

            feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            
            print(f"\n--- Top 10 Features by {importance_type} ---")
            print(feature_importance.head(10).to_markdown(numalign="left", stralign="left"))
            
        except Exception as e:
            self.logger.error(f"Could not extract feature importance: {e}")
            print(f"\nCould not extract feature importance: {e}")

    def save_model(self, file_path: str):
        """
        Saves the current model (Supervised or Unsupervised) to disk.
        Args: file_path (str): Destination path for the model file.
        Raises: DataAnalyzerError: If saving fails.
        """
        try:
            save_object = {
                'target_col': self.target_col,
                'features_list': self.features_list,
                'model_type': self.model_type
            }
            
            if self.model_pipeline is not None:
                save_object['pipeline'] = self.model_pipeline
                save_object['type'] = 'supervised'
            elif self.unsupervised_model is not None:
                save_object['pipeline'] = self.unsupervised_model # Contains preprocessor + model
                save_object['type'] = 'unsupervised'
                save_object['unsupervised_type'] = self.unsupervised_type
            else:
                raise ValueError("No trained model to save.")

            joblib.dump(save_object, file_path)
            self.logger.info(f"Model saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise DataAnalyzerError(f"Failed to save model: {e}") from e

    def load_model(self, file_path: str):
        """
        Loads a saved model from disk.
        Args: file_path (str): Path to the model file.
        Raises: DataAnalyzerError: If loading fails.
        """
        try:
            if not os.path.exists(file_path): raise FileNotFoundError(f"Model file not found: {file_path}")
            
            # SECURITY WARNING
            print("\n[SECURITY WARNING] Loading models with joblib is insecure if the file is untrusted.")
            print("Only load models you created or trust implicitly.\n")
            
            loaded_object = joblib.load(file_path)
            
            if loaded_object.get('type') == 'unsupervised':
                self.unsupervised_model = loaded_object['pipeline']
                self.unsupervised_type = loaded_object.get('unsupervised_type')
                self.model_type = 'unsupervised'
                print(f"Unsupervised model loaded. Type: {self.unsupervised_type}")
            else:
                self.model_pipeline = loaded_object.get('pipeline')
                self.target_col = loaded_object.get('target_col')
                self.features_list = loaded_object.get('features_list')
                self.model_type = loaded_object.get('model_type')
                print(f"Supervised model loaded. Type: {self.model_type}, Target: {self.target_col}")
                
            self.logger.info(f"Successfully loaded model from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise DataAnalyzerError(f"Failed to load model: {e}") from e

    def predict_new_data(self, new_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for new data using the trained model.
        Args: new_data_df (pd.DataFrame): Dataframe containing features matching the training data.
        Returns: pd.DataFrame: Original dataframe with a 'prediction' column appended.
        Raises: DataAnalyzerError: If prediction fails or model not trained.
        """
        if self.model_pipeline is None: raise DataAnalyzerError("No model is loaded or trained.")
        
        try:
            missing_cols = set(self.features_list) - set(new_data_df.columns)
            if missing_cols:
                raise ValueError(f"New data is missing required columns: {missing_cols}")
                
            X_new = new_data_df[self.features_list]
            predictions = self.model_pipeline.predict(X_new)
            
            results_df = new_data_df.copy()
            results_df['prediction'] = predictions
            
            self.logger.info(f"Successfully made {len(predictions)} predictions.")
            return results_df
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
            
    def query_data(self, query: str) -> pd.DataFrame:
        if self.df is None: 
            raise ValueError("Data not loaded.")

        try:
            result_df = self.df.query(query)
            self.logger.info(f"Query executed. Result shape: {result_df.shape}")
            return result_df
        except Exception as e:
            self.logger.error(f"Pandas query failed: {str(e)}")
            raise ValueError(f"Invalid query. Use Pandas query syntax. Details: {e}")

    def plot_data(self, plot_type: str, column_x: str, column_y: str = None):
        if self.df is None: 
            raise ValueError("Data not loaded.")
        plt.figure(figsize=(10, 6))
        try:
            if plot_type == 'histogram':
                if column_x not in self.df.columns: 
                    raise ValueError(f"Column '{column_x}' not found.")
                if not is_numeric_dtype(self.df[column_x]): 
                    raise ValueError(f"Column '{column_x}' is not numeric.")
                self.df[column_x].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7)
                plt.title(f'Histogram of {column_x}')
            
            elif plot_type == 'scatter':
                if column_x not in self.df.columns or column_y not in self.df.columns: 
                    raise ValueError(f"Column(s) not found.")
                if not is_numeric_dtype(self.df[column_x]) or not is_numeric_dtype(self.df[column_y]):
                    raise ValueError(f"Both columns must be numeric.")
                plt.scatter(self.df[column_x], self.df[column_y], alpha=0.7)
                plt.title(f'Scatter Plot of {column_x} vs {column_y}')

            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show() 
            self.logger.info(f"Generated {plot_type} plot.")
        except Exception as e:
            self.logger.error(f"Plotting failed: {str(e)}")
            raise

    def get_analysis_options(self) -> dict:
        """Get summary of data columns."""
        if self.df is None: 
            raise ValueError("Data not loaded.")
        
        types = self.get_column_types()
        return {
            'numeric_columns': types['numeric'],
            'categorical_columns': types['categorical'],
            'datetime_columns': types['datetime'],
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'all_columns': self.df.columns.tolist()
        }
    
    def run_unsupervised_analysis(self, analysis_type: str, k_or_n: int):
        """
        Runs K-Means Clustering or PCA on the currently loaded data.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Please load a file first (Option 1).")

        if not isinstance(k_or_n, int) or k_or_n < 1:
            raise ValueError("K or N must be a positive integer greater than 0.")
        
        self.logger.info(f"Starting unsupervised analysis: {analysis_type} with parameter {k_or_n}")

        # Separate features from the existing DataFrame
        column_types = self.get_column_types()
        
        # Use all available features for unsupervised learning
        numeric_features = column_types['numeric']
        categorical_features = column_types['categorical']
        datetime_features = column_types['datetime']
        
        all_features = numeric_features + categorical_features + datetime_features
        if not all_features:
            raise ValueError("No valid features found after cleaning. Cannot run unsupervised analysis.")
        
        X = self.df[all_features]
        
        # Create the full preprocessor (scaling and OHE)
        preprocessor = create_preprocessor(numeric_features, categorical_features, datetime_features)
        
        try:
            # 1. Fit and transform the entire dataset (no train/test split needed for UL)
            X_processed = preprocessor.fit_transform(X)
            
            if analysis_type == 'clustering':
                if k_or_n > len(X):
                     raise ValueError(f"K ({k_or_n}) cannot be greater than the number of samples ({len(X)}).")
                
                # Use standard KMeans
                model = KMeans(n_clusters=k_or_n, random_state=42, n_init=10, max_iter=300)
                labels = model.fit_predict(X_processed)
                
                # Add the labels back to the DataFrame
                new_col_name = f'cluster_k{k_or_n}'
                self.df[new_col_name] = labels
                
                self.logger.info(f"K-Means Clustering complete. Added column: '{new_col_name}'")
                self.logger.info(f"\n--- Clustering Analysis Complete ---")
                self.logger.info(f"Added column: '{new_col_name}' to the DataFrame.")
                self.logger.info(f"Cluster sizes:\n{self.df[new_col_name].value_counts().to_markdown()}")
            
            elif analysis_type == 'pca':
                if k_or_n > X_processed.shape[1]:
                    raise ValueError(f"N ({k_or_n}) cannot be greater than the number of features after preprocessing ({X_processed.shape[1]}).")
                
                model = PCA(n_components=k_or_n, random_state=42)
                components = model.fit_transform(X_processed)
                
                component_names = [f'pca_c{i+1}' for i in range(k_or_n)]
                for i, name in enumerate(component_names):
                    self.df[name] = components[:, i]
                
                explained_variance = model.explained_variance_ratio_.sum()
                
                self.logger.info(f"PCA complete. Added {k_or_n} components.")
                print(f"\n--- PCA Analysis Complete ---")
                print(f"Added columns: {', '.join(component_names)}")
                print(f"Total variance explained by {k_or_n} components: {explained_variance:.4f}")

            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")

            # Save the unsupervised pipeline (preprocessor + model)
            self.unsupervised_model = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            self.unsupervised_type = analysis_type

        except Exception as e:
            self.logger.error(f"Unsupervised analysis failed: {str(e)}")
            raise

    def run_forecasting(self, target_col: str, date_col: str, periods: int = 30) -> dict:
        """
        Runs Time Series Forecasting using Prophet.
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Please install it using 'pip install prophet'.")

        if self.df is None:
            raise ValueError("Data not loaded.")

        if target_col not in self.df.columns or date_col not in self.df.columns:
            raise ValueError(f"Columns {target_col} and/or {date_col} not found in data.")

        self.logger.info(f"\n--- Starting Prophet Forecasting ---")
        self.logger.info(f"Forecasting target '{target_col}' using date column '{date_col}' for {periods} periods.")

        try:
            prophet_df = self.df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})    
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
            prophet_df = prophet_df.dropna()
            
            model = Prophet()
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            fig1 = model.plot(forecast)
            fig2 = model.plot_components(forecast)
            
            self.logger.info("Forecasting complete.")
            
            return {
                'forecast': forecast,
                'model': model,
                'figures': [fig1, fig2],
                'forecast_tail': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            }
            
        except Exception as e:
            self.logger.error(f"Forecasting failed: {str(e)}")
            raise

    def train_rl_agent(self, target_col: str, total_timesteps: int = 10000, save_path: str = "rl_trend_predictor"):
        """
        Trains a PPO (Proximal Policy Optimization) Agent to predict trends.
        """
        if not RL_AVAILABLE:
            self.logger.error("RL libraries not installed. Please install 'stable-baselines3' and 'gymnasium'.")
            print("Error: RL libraries missing. Install stable-baselines3 and gymnasium.")
            return

        if self.df is None:
            raise ValueError("Data not loaded.")
        
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        # We need numeric targets for trend prediction
        if not is_numeric_dtype(self.df[target_col]):
            raise ValueError("RL Trend Prediction requires a numeric target column.")

        self.logger.info(f"\n--- Starting RL Training (PPO) ---")
        self.logger.info(f"Target: {target_col}")
        self.logger.info(f"Timesteps: {total_timesteps}")
        self.logger.info(f"Initializing RL environment for target: {target_col}")

        try:
            # 1. Preprocess Data specifically for RL (Need pure numeric matrix)
            # We reuse the helper to get a clean, numeric-only dataframe for the environment
            column_types = self.get_column_types()
            numeric_features = column_types['numeric']
            
            # Use ONLY numeric columns for the RL state to keep it simple/stable
            df_rl = self.df[numeric_features].copy()
            
            # Fill any remaining NaNs
            df_rl = df_rl.fillna(0)

            # 2. Create the Environment
            # We wrap it in DummyVecEnv as required by Stable Baselines
            env = DummyVecEnv([lambda: DataFrameEnv(df_rl, target_col)])

            # 3. Initialize the Agent (PPO is robust and general-purpose)
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)

            # 4. Train
            print("Training in progress... (This may take a moment)")
            model.learn(total_timesteps=total_timesteps)
            
            self.logger.info("\n--- RL Training Complete ---")
            
            # 5. Simple Evaluation Loop
            obs = env.reset()
            total_reward = 0
            action_counts = {0:0, 1:0}
            
            # Run one 'episode' (one pass through the data)
            print("Evaluating Agent on full dataset...")
            for _ in range(len(df_rl) - 1):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                total_reward += rewards[0]
                action_counts[int(action[0])] += 1
                if dones[0]:
                    break
            
            print(f"Total Reward Acquired: {total_reward:.2f}")
            print(f"Actions taken: {action_counts} (0=Decrease/Hold, 1=Increase)")
            
            # Save the RL model separately
            model.save(save_path)
            print(f"Agent saved to '{save_path}.zip'")
            self.logger.info("RL Agent trained and saved.")
            
            # MEMORY: Force garbage collection
            del model
            del env
            gc.collect()

        except Exception as e:
            self.logger.error(f"RL Training failed: {e}")
            raise