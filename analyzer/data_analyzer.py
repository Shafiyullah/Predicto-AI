import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import warnings
import re
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from analyzer.preprocessing import create_preprocessor, DatetimeFeatureExtractor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

class Data_Analyzer:
    def __init__(self):
        self.df = None
        self.model_pipeline = None
        self.target_col = None
        self.features_list = None
        self.model_type = None
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
            raise
    
    def load_new_dataframe(self, file_path: str) -> pd.DataFrame:
        self.validate_file_path(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            # MEMORY FIX: Added low_memory=True
            df = pd.read_csv(file_path, on_bad_lines='warn', encoding='utf-8', low_memory=True)
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
        for col in df.columns:
            if not is_numeric_dtype(df[col]) and not is_datetime64_any_dtype(df[col]):
                # Attempt numeric conversion
                converted_numeric = pd.to_numeric(df[col], errors='coerce')
                if converted_numeric.notnull().sum() > 0.8 * len(df):
                     df[col] = converted_numeric
                     self.logger.info(f"Converted column '{col}' to numeric.")
                     continue
                
                try:
                    converted_datetime = pd.to_datetime(df[col], errors='coerce')
                    # If >80% are valid dates, convert the column
                    if converted_datetime.notnull().sum() > 0.8 * len(df):
                        df[col] = converted_datetime
                        self.logger.info(f"Converted column '{col}' to datetime.")
                        continue
                except Exception:
                    pass

        # 3. Sanitize string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(r'^[<>"\';\n\r]+$', '', regex=True)
        
        return df
    
    def _downcast_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        """ MEMORY : Downcast numeric columns to save memory."""
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if 'int' in str(df[col].dtype):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif 'float' in str(df[col].dtype):
                df[col] = pd.to_numeric(df[col], downcast='float')
        self.logger.info("Numeric columns downcasted for memory efficiency.")
        return df

    def load_data(self, file_path: str):
        try:
            df = self.load_new_dataframe(file_path)
            df = self._prepare_data_for_processing(df)
            df = self._downcast_numerics(df)
            
            # 4. Drop duplicates
            original_shape = df.shape
            df = df.drop_duplicates()
            cleaned_shape = df.shape
            self.logger.info(f"Dropped {original_shape[0] - cleaned_shape[0]} duplicate rows.")
            
            self.df = df
            self.logger.info(f"Data loaded and prepared: {len(df)} rows, {len(df.columns)} columns.")
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise

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
        if self.df is None: raise ValueError("Data not loaded.")
        if target_col not in self.df.columns: raise ValueError(f"Target column '{target_col}' not found.")
            
        self.target_col = target_col
        
        # 1. Identify Features (X) and Target (y)
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
            
        # 2. Split Data FIRST
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Create Preprocessing Pipeline
        preprocessor = create_preprocessor(numeric_features, categorical_features, datetime_features)
        
        # 4. Define Models and Hyperparameter Grids 
        if self.model_type == 'regression':
            models = {
                'LinearRegression': (LinearRegression(), {}),
                'RandomForest': (RandomForestRegressor(random_state=42), 
                                 {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, 20, None]}),
                'GradientBoosting': (GradientBoostingRegressor(random_state=42), 
                                     {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]})
            }
            scoring = 'r2'
        else:
            models = {
                'LogisticRegression': (LogisticRegression(random_state=42, max_iter=1000), 
                                       {'model__C': [0.1, 1.0, 10]}),
                'RandomForest': (RandomForestClassifier(random_state=42), 
                                 {'model__n_estimators': [50, 100, 200], 'model__max_depth': [5, 10, 20, None]}),
                'GradientBoosting': (GradientBoostingClassifier(random_state=42), 
                                     {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]})
            }
            scoring = 'accuracy'
            
        best_score = -np.inf
        best_pipeline = None
        best_model_name = ""

        # 5. Train and Evaluate Models using RandomizedSearchCV 
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
            
        self.model_pipeline = best_pipeline
        self.logger.info(f"Selected best model: {best_model_name} with CV score: {best_score:.4f}")
        
        # 6. Final Evaluation & Feature Importance
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

    def show_feature_importance(self):
        """ Extracts and displays feature importance from the trained pipeline. """
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
        if self.model_pipeline is None: raise ValueError("No model has been trained.")
        try:
            save_object = {
                'pipeline': self.model_pipeline,
                'target_col': self.target_col,
                'features_list': self.features_list,
                'model_type': self.model_type
            }
            joblib.dump(save_object, file_path)
            self.logger.info(f"Model pipeline saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, file_path: str):
        try:
            if not os.path.exists(file_path): raise FileNotFoundError(f"Model file not found: {file_path}")
            loaded_object = joblib.load(file_path)
            self.model_pipeline = loaded_object['pipeline']
            self.target_col = loaded_object['target_col']
            self.features_list = loaded_object['features_list']
            self.model_type = loaded_object['model_type']
            self.logger.info(f"Successfully loaded model from {file_path}")
            print(f"Model loaded. Type: {self.model_type}, Target: {self.target_col}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_new_data(self, new_data_df: pd.DataFrame) -> pd.DataFrame:
        if self.model_pipeline is None: raise ValueError("No model is loaded or trained.")
        
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
        if self.df is None: raise ValueError("Data not loaded.")
        try:
            result_df = self.df.query(query)
            self.logger.info(f"Query executed. Result shape: {result_df.shape}")
            return result_df
        except Exception as e:
            self.logger.error(f"Pandas query failed: {str(e)}")
            raise ValueError(f"Invalid query. Use Pandas query syntax. Details: {e}")

    def plot_data(self, plot_type: str, column_x: str, column_y: str = None):
        if self.df is None: raise ValueError("Data not loaded.")
        plt.figure(figsize=(10, 6))
        try:
            if plot_type == 'histogram':
                if column_x not in self.df.columns: raise ValueError(f"Column '{column_x}' not found.")
                if not is_numeric_dtype(self.df[column_x]): raise ValueError(f"Column '{column_x}' is not numeric.")
                self.df[column_x].plot(kind='hist', bins=30, edgecolor='black', alpha=0.7)
                plt.title(f'Histogram of {column_x}')
            
            elif plot_type == 'scatter':
                if column_x not in self.df.columns or column_y not in self.df.columns: raise ValueError(f"Column(s) not found.")
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
        if self.df is None: raise ValueError("Data not loaded.")
        
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
        # Note: We use the existing preprocessor pipeline to ensure data is scaled and encoded.
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
                print(f"\n--- Clustering Analysis Complete ---")
                print(f"Added column: '{new_col_name}' to the DataFrame.")
                print(f"Cluster sizes:\n{self.df[new_col_name].value_counts().to_markdown()}")
            
            elif analysis_type == 'pca':
                if k_or_n > X_processed.shape[1]:
                    raise ValueError(f"N ({k_or_n}) cannot be greater than the number of features after preprocessing ({X_processed.shape[1]}).")
                    
                # Run PCA
                model = PCA(n_components=k_or_n, random_state=42)
                components = model.fit_transform(X_processed)
                
                # Add components back to the DataFrame
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

        except Exception as e:
            self.logger.error(f"Unsupervised analysis failed: {str(e)}")
            raise