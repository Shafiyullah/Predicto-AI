import sys
from analyzer.data_analyzer import DataAnalyzer

def main():
    analyzer = DataAnalyzer()
    print("=== Secure Data Analyzer ===")
    
    while True:
        try:
            print("\n--- Menu ---")
            print("1. Load, clean, and optimize file (CSV, XLSX, JSON)")
            print("2. View data info")
            print("3. Train a new model")
            print("4. Run Unsupervised Analysis")
            print("5. Train RL Agent")
            print("6. Save the current model")
            print("7. Load a saved model")
            print("8. Predict on new data (file)")
            print("9. Run Data Query (Pandas)")
            print("10. Visualize Data (Plotting)")
            print("0. Exit")
            
            choice = input("Choose option (0-10): ").strip()
            
            if choice == '1':
                file_path = input("Enter file path: ").strip()
                analyzer.load_data(file_path)
                print(f"Successfully loaded and optimized: {analyzer.df.shape[0]} rows, {analyzer.df.shape[1]} columns.")
                
            elif choice == '2':
                if analyzer.df is None:
                    analyzer.logger.info("Load data first (Option 1).")
                    continue

                options = analyzer.get_analysis_options()
                analyzer.logger.info(f"\nTotal Rows: {options['total_rows']}")
                analyzer.logger.info(f"Total Columns: {options['total_columns']}")
                analyzer.logger.info(f"\nAll Columns:\n {options['all_columns']}")
                analyzer.logger.info(f"\nValid Numeric Columns:\n {options['numeric_columns']}")
                analyzer.logger.info(f"\nValid Categorical Columns (<=100 unique):\n {options['categorical_columns']}") 
                analyzer.logger.info(f"\nDetected Datetime Columns:\n {options['datetime_columns']}") 
                
            elif choice == '3':
                if analyzer.df is None:
                    analyzer.logger.info("Load data first (Option 1).")
                    continue

                options = analyzer.get_analysis_options()
                analyzer.logger.info("\n--- Train New Model ---")
                analyzer.logger.info(f"All available columns: {options['all_columns']}")
                target = input("Enter the EXACT target column name to predict: ").strip()
                analyzer.logger.info(f"Starting model training for target: {target}...")
                analyzer.train_model(target)
            
            # --- Unsupervised Analysis ---
            elif choice == '4':
                if analyzer.df is None:
                    analyzer.logger.info("Load data first (Option 1).")
                    continue
                
                analyzer.logger.info("\n--- Unsupervised Analysis ---")
                analyzer.logger.info("1. Clustering (K-Means)")
                analyzer.logger.info("2. Dimensionality Reduction (PCA)")
                
                analysis_choice = input("Select analysis type (1 or 2): ").strip()
                
                if analysis_choice == '1':
                    analysis_type = 'clustering'
                    try:
                        k = int(input("Enter number of clusters (K): ").strip())
                        analyzer.run_unsupervised_analysis(analysis_type, k)
                    except ValueError as e:
                        print(f"Input Error: {e}")
                        analyzer.logger.warning(f"Unsupervised Analysis Input Error: {e}")
                    
                elif analysis_choice == '2':
                    analysis_type = 'pca'
                    try:
                        n = int(input("Enter number of components (N): ").strip())
                        analyzer.run_unsupervised_analysis(analysis_type, n)
                    except ValueError as e:
                        print(f"Input Error: {e}")
                        analyzer.logger.warning(f"PCA Input Error: {e}")
                
                else:
                    print("Invalid analysis choice.")
            
            # --- Reinforcement Learning Analysis ---
            elif choice == '5':
                if analyzer.df is None:
                    print("Load data first (Option 1).")
                    continue
            
                options = analyzer.get_analysis_options()
                print(f"\nAvailable numeric targets: {options['numeric_columns']}")
                target = input("Enter numeric target column for Trend Prediction: ").strip()
            
                try:
                    steps = input("Enter training timesteps (default 10000): ").strip()
                    steps = int(steps) if steps else 10000
                    save_path = input("Enter path to save RL agent (default: rl_trend_predictor): ").strip()
                    if not save_path:
                        save_path = "rl_trend_predictor"
                    
                    analyzer.train_rl_agent(target, total_timesteps=steps, save_path=save_path)
                except Exception as e:
                    print(f"RL Error: {e}")
                    analyzer.logger.error(f"RL Training failed: {e}")
            
            elif choice == '6':
                if analyzer.model_pipeline is None and analyzer.unsupervised_model is None:
                    print("No model has been trained yet (Option 3, 4, or 5).")
                    continue

                file_path = input("Enter file path to save model (e.g., my_model.joblib): ").strip()
                analyzer.save_model(file_path)
                print(f"Model saved to {file_path}")

            elif choice == '7':
                file_path = input("Enter file path to load model from (e.g., my_model.joblib): ").strip()
                analyzer.load_model(file_path)

            elif choice == '8':
                if analyzer.model_pipeline is None:
                    print("No model is loaded (Option 5) or trained (Option 3).")
                    continue
                
                file_path = input("Enter path to NEW data file for prediction: ").strip()

                try:
                    new_df = analyzer.load_new_dataframe(file_path)
                    clean_new_df = analyzer._prepare_data_for_processing(new_df)
                except Exception as e:
                    print(f"Error loading new data: {e}")
                    analyzer.logger.error(f"Prediction data load failed: {e}")
                    continue

                results_df = analyzer.predict_new_data(clean_new_df)
                print("\n--- Predictions (Head) ---")
                print(results_df.head(10).to_markdown(index=False))
                
                save_choice = input("Save prediction results to CSV? (y/n): ").strip().lower()
                if save_choice == 'y':
                    results_df.to_csv("predictions.csv", index=False)
                    print("Saved to predictions.csv")
            
            elif choice == '9':
                if analyzer.df is None:
                    print("Load data first (Option 1).")
                    continue
                print("\n**DataFrame is available. Use Pandas query syntax.**")
                print("Example: salary > 50000 and age < 30")
                print(f"Available columns: {analyzer.df.columns.tolist()}")
                query = input("Enter query: ").strip()
                result_df = analyzer.query_data(query)
                
                print("\n--- Query Results (Head) ---")
                if not result_df.empty:
                    analyzer.logger.info(result_df.head(10).to_markdown(index=False))
                    analyzer.logger.info(f"\nTotal rows returned: {len(result_df)}")
                else:
                    analyzer.logger.info("Query returned an empty result set.")

            elif choice == '10':
                if analyzer.df is None:
                    analyzer.logger.info("Load data first (Option 1).")
                    continue
                options = analyzer.get_analysis_options()
                analyzer.logger.info(f"\nAvailable numeric columns: {options['numeric_columns']}")
                analyzer.logger.info("\nChoose plot type:")
                analyzer.logger.info("  1. Histogram (single numeric column)")
                analyzer.logger.info("  2. Scatter Plot (two numeric columns)")
                plot_choice = input("Enter plot type (1 or 2): ").strip()

                if plot_choice == '1':
                    col_x = input("Enter column name for histogram: ").strip()
                    analyzer.plot_data('histogram', column_x=col_x)
                elif plot_choice == '2':
                    col_x = input("Enter X-axis column name: ").strip()
                    col_y = input("Enter Y-axis column name: ").strip()
                    analyzer.plot_data('scatter', column_x=col_x, column_y=col_y)
                else:
                    print("Invalid plot choice.")
                
            elif choice == '0':
                print("Exiting application. Goodbye!")
                sys.exit()
                
            else:
                print("Invalid choice! Please choose from 1-9.")
                
        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            sys.exit()
        except Exception as e:
            analyzer.logger.error(f"Top-level error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()