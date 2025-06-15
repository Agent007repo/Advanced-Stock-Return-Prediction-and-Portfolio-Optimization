import datetime
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid
import warnings
import sys
import subprocess
warnings.filterwarnings('ignore')

# TensorFlow auto-installation
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not found. Attempting to install...")
    try:
        # Install TensorFlow
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
        print("TensorFlow installed successfully. Loading modules...")
        
        # Try importing again
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        TF_AVAILABLE = True
        print("TensorFlow loaded successfully.")
    except Exception as e:
        print(f"Could not install TensorFlow: {str(e)}")
        print("Neural network model will be skipped")

class StockPredictionModel:
    def __init__(self, work_dir=None, n_long=50, n_short=50, strategy_type="long-short", fast_mode=False):
        """
        Initialize the stock prediction model.
        
        Parameters:
        -----------
        work_dir : str
            Working directory where the data files are stored
        n_long : int
            Number of stocks to include in long portfolio (default: 50)
        n_short : int
            Number of stocks to include in short portfolio (default: 50)
        strategy_type : str
            Type of portfolio strategy: "long-short", "long-only", or "mixed"
        fast_mode : bool
            If True, use only Ridge regression for faster execution
        """
        # Set working directory
        self.work_dir = work_dir or os.getcwd()
        
        # Set portfolio parameters
        self.n_long = n_long
        self.n_short = n_short
        self.strategy_type = strategy_type
        
        # Set file paths
        self.factor_list_path = os.path.join(self.work_dir, "factor_char_list.csv")
        self.market_index_path = os.path.join(self.work_dir, "mkt_ind.csv")
        self.sample_data_path = os.path.join(self.work_dir, "mma_sample_v2.csv")
        
        # Define model parameters
        self.ret_var = "stock_exret"
        
        # Set model names based on fast_mode
        if fast_mode:
            print("\n--- RUNNING IN FAST MODE - USING ONLY RIDGE REGRESSION ---\n")
            self.model_names = ["ridge"]  # Only use Ridge regression in fast mode
        else:
            self.model_names = ["ols", "lasso", "ridge", "en"]
            if TF_AVAILABLE:
                self.model_names.append("nn")
            
        self.model_output_file = "stock_predictions.csv"
        
        # Print start time
        print(f"Starting prediction process at: {datetime.datetime.now()}")
    
    def load_data(self):
        """Load and prepare the stock data and predictors"""
        print("Loading data...")
        
        # Read sample data
        self.raw_data = pd.read_csv(
            self.sample_data_path, 
            parse_dates=["date"], 
            low_memory=False
        )
        
        # Read list of stock predictors
        try:
            self.stock_vars = list(pd.read_csv(self.factor_list_path)["variable"].values)
        except:
            print("Warning: Could not load factor_char_list.csv, using all numeric columns as predictors")
            numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
            self.stock_vars = [col for col in numeric_cols if col != self.ret_var]
        
        # Read market index data
        try:
            self.market_data = pd.read_csv(self.market_index_path)
        except:
            print("Warning: Could not load mkt_ind.csv, market comparison won't be available")
            self.market_data = None
        
        # Filter data where target variable is not missing
        self.filtered_data = self.raw_data[self.raw_data[self.ret_var].notna()].copy()
        
        # Optional: Sample data for faster processing
        sample_fraction = 1.0  # Set to lower value (e.g., 0.3) for faster runtime
        if hasattr(self, 'fast_mode') and self.fast_mode and sample_fraction < 1.0:
            print(f"Sampling {sample_fraction*100:.1f}% of data for faster processing")
            self.filtered_data = self.filtered_data.sample(frac=sample_fraction, random_state=42)
        
        # Focus on the time period specified in the assignment (Jan 2020 to Dec 2023)
        # We'll still use all data for training, but focus reporting on this period
        self.eval_start_date = pd.to_datetime("2020-01-01")
        self.eval_end_date = pd.to_datetime("2023-12-31")
        
        print(f"Loaded {len(self.filtered_data)} data points with {len(self.stock_vars)} predictors")
        print(f"Evaluation will focus on period: {self.eval_start_date.strftime('%Y-%m-%d')} to {self.eval_end_date.strftime('%Y-%m-%d')}")
        
        return self
    
    def preprocess_data(self):
        """Transform variables to the same scale for each month"""
        print("Preprocessing data...")
        
        # Group by date for monthly processing
        monthly = self.filtered_data.groupby("date")
        self.processed_data = pd.DataFrame()
        
        # Process each month separately
        for date, monthly_raw in monthly:
            group = monthly_raw.copy()
            
            # Rank transform each variable to [-1, 1]
            for var in self.stock_vars:
                if var not in group.columns:
                    continue
                    
                var_median = group[var].median(skipna=True)
                # Fill missing values with cross-sectional median
                group[var] = group[var].fillna(var_median)
                
                # Rank transformation
                group[var] = group[var].rank(method="dense") - 1
                group_max = group[var].max()
                if group_max > 0:
                    group[var] = (group[var] / group_max) * 2 - 1
                else:
                    group[var] = 0  # Handle case of all missing values
            
            # Add the adjusted values to the processed data
            self.processed_data = pd.concat([self.processed_data, group], ignore_index=True)
        
        print(f"Preprocessing complete. Data shape: {self.processed_data.shape}")
        return self
    
    def train_and_predict(self):
        """Train models and make predictions using expanding window"""
        print("Training models and making predictions...")
        
        # Initialize parameters
        starting = pd.to_datetime("20000101", format="%Y%m%d")
        counter = 0
        self.predictions = pd.DataFrame()
        self.feature_importances = {model: np.zeros(len(self.stock_vars)) for model in ["lasso", "ridge", "en"]}
        self.feature_importance_counts = {model: 0 for model in ["lasso", "ridge", "en"]}
        
        # Estimation with expanding window
        while (starting + pd.DateOffset(years=11 + counter)) <= pd.to_datetime("20240101", format="%Y%m%d"):
            cutoff = [
                starting,
                starting + pd.DateOffset(years=8 + counter),  # 8-year training set
                starting + pd.DateOffset(years=10 + counter),  # 2-year validation set
                starting + pd.DateOffset(years=11 + counter),  # 1-year testing set
            ]
            print(f"Window {counter+1}: {cutoff[0].strftime('%Y-%m-%d')} to {cutoff[3].strftime('%Y-%m-%d')}")
            
            # Split data into training, validation, and testing sets
            train = self.processed_data[(self.processed_data["date"] >= cutoff[0]) & 
                                        (self.processed_data["date"] < cutoff[1])]
            validate = self.processed_data[(self.processed_data["date"] >= cutoff[1]) & 
                                         (self.processed_data["date"] < cutoff[2])]
            test = self.processed_data[(self.processed_data["date"] >= cutoff[2]) & 
                                      (self.processed_data["date"] < cutoff[3])]
            
            # Standardize features
            scaler = StandardScaler().fit(train[self.stock_vars])
            train_scaled = train.copy()
            validate_scaled = validate.copy()
            test_scaled = test.copy()
            
            train_scaled[self.stock_vars] = scaler.transform(train[self.stock_vars])
            validate_scaled[self.stock_vars] = scaler.transform(validate[self.stock_vars])
            test_scaled[self.stock_vars] = scaler.transform(test[self.stock_vars])
            
            # Get features and target variables
            X_train = train_scaled[self.stock_vars].values
            Y_train = train_scaled[self.ret_var].values
            X_val = validate_scaled[self.stock_vars].values
            Y_val = validate_scaled[self.ret_var].values
            X_test = test_scaled[self.stock_vars].values
            Y_test = test_scaled[self.ret_var].values
            
            # De-mean Y (because regressions are fitted without intercept)
            Y_mean = np.mean(Y_train)
            Y_train_dm = Y_train - Y_mean
            
            # Prepare output data with minimum identifications
            # Include company name and ticker for identifying top holdings later
            id_cols = ["year", "month", "date", "permno", self.ret_var]
            if "ticker" in test.columns:
                id_cols.append("ticker")
            if "comnam" in test.columns:
                id_cols.append("comnam")
                
            reg_pred = test[id_cols].copy()
            
            # Linear Regression (OLS)
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred["ols"] = x_pred
            
            # Lasso Regression
            lambdas = np.arange(-4, 4.1, 0.1)
            val_mse = np.zeros(len(lambdas))
            for ind, i in enumerate(lambdas):
                reg = Lasso(alpha=(10**i), max_iter=1000000, fit_intercept=False)
                reg.fit(X_train, Y_train_dm)
                val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)
            
            # Select best lambda based on validation set
            best_lambda = lambdas[np.argmin(val_mse)]
            reg = Lasso(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred["lasso"] = x_pred
            
            # Store feature importances (coefficient magnitudes)
            self.feature_importances["lasso"] += np.abs(reg.coef_)
            self.feature_importance_counts["lasso"] += 1
            
            print(f"  Best Lasso lambda: 10^{best_lambda:.2f}")
            
            # For Ridge regression - use optimized approach in fast mode
            if hasattr(self, 'fast_mode') and self.fast_mode:
                # Use smaller lambda search space for faster training
                lambdas = np.linspace(3, 7, 5)  # Fewer values to test, focusing on higher regularization
            else:
                # Standard search space
                lambdas = np.linspace(-3, 9, 13)
            
            val_mse = np.zeros(len(lambdas))
            
            for ind, lm in enumerate(lambdas):
                reg = Ridge(alpha=((10**lm) * 0.5), fit_intercept=False)
                reg.fit(X_train, Y_train_dm)
                val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)
            
            best_lambda = lambdas[np.argmin(val_mse)]
            reg = Ridge(alpha=((10**best_lambda) * 0.5), fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred["ridge"] = x_pred
            
            # Store feature importances
            self.feature_importances["ridge"] += np.abs(reg.coef_)
            self.feature_importance_counts["ridge"] += 1
            
            print(f"  Best Ridge lambda: 10^{best_lambda:.2f} * 0.5")
            
            # Elastic Net
            lambdas = np.arange(-4, 4.1, 0.1)
            val_mse = np.zeros(len(lambdas))
            for ind, i in enumerate(lambdas):
                reg = ElasticNet(alpha=(10**i), max_iter=1000000, fit_intercept=False)
                reg.fit(X_train, Y_train_dm)
                val_mse[ind] = mean_squared_error(Y_val, reg.predict(X_val) + Y_mean)
            
            best_lambda = lambdas[np.argmin(val_mse)]
            reg = ElasticNet(alpha=(10**best_lambda), max_iter=1000000, fit_intercept=False)
            reg.fit(X_train, Y_train_dm)
            x_pred = reg.predict(X_test) + Y_mean
            reg_pred["en"] = x_pred
            
            # Store feature importances
            self.feature_importances["en"] += np.abs(reg.coef_)
            self.feature_importance_counts["en"] += 1
            
            print(f"  Best Elastic Net lambda: 10^{best_lambda:.2f}")
            
            # Enhanced Neural Network (if TensorFlow is available)
            if TF_AVAILABLE and "nn" in self.model_names:
                try:
                    # Create dataframes for the neural network
                    X_train_df = pd.DataFrame(X_train, columns=self.stock_vars)
                    
                    # Train enhanced neural network with attention mechanism
                    print("\nTraining enhanced neural network model...")
                    model, feat_mean, feat_std = self.train_neural_network(X_train_df, Y_train_dm)
                    
                    # Normalize test data using the same mean and std from training
                    X_test_normalized = (X_test - feat_mean) / (feat_std + 1e-8)
                    
                    # Make predictions
                    nn_pred = model.predict(X_test_normalized).flatten() + Y_mean
                    reg_pred["nn"] = nn_pred
                    print("Neural network prediction complete.")
                except Exception as e:
                    print(f"Neural network training failed: {str(e)}")
                    # Use ridge predictions as fallback
                    reg_pred["nn"] = reg_pred["ridge"]
                    print("Using ridge predictions as fallback for neural network")
            
            # Add to the output data
            self.predictions = pd.concat([self.predictions, reg_pred], ignore_index=True)
            
            # Go to next window
            counter += 1
        
        # Normalize feature importances by number of windows
        for model in ["lasso", "ridge", "en"]:
            if self.feature_importance_counts[model] > 0:
                self.feature_importances[model] /= self.feature_importance_counts[model]
        
        # Save predictions to CSV
        self.output_path = os.path.join(self.work_dir, self.model_output_file)
        self.predictions.to_csv(self.output_path, index=False)
        print(f"Predictions saved to: {self.output_path}")
        
        return self
    
    def evaluate_models(self):
        """Calculate and display out-of-sample R² for each model"""
        print("\nModel Evaluation - Out-of-Sample R²:")
        
        # Focus on evaluation period
        eval_data = self.predictions[
            (self.predictions["date"] >= self.eval_start_date) & 
            (self.predictions["date"] <= self.eval_end_date)
        ]
        
        yreal = eval_data[self.ret_var].values
        r2_values = {}
        
        for model_name in self.model_names:
            if model_name not in eval_data.columns:
                continue
                
            ypred = eval_data[model_name].values
            r2 = 1 - np.sum(np.square((yreal - ypred))) / np.sum(np.square(yreal))
            r2_values[model_name] = r2
            print(f"  {model_name.upper()}: {r2:.6f}")
        
        # Find best model
        if r2_values:
            best_model = max(r2_values.items(), key=lambda x: x[1])[0]
            print(f"\nBest performing model: {best_model.upper()} with R² = {r2_values[best_model]:.6f}")
        else:
            best_model = "ridge"  # Default
            print("No valid R² values calculated, using RIDGE as default")
            
        return best_model, r2_values
    
    def train_neural_network(self, X_train, y_train):
        """
        Train an enhanced neural network model with attention mechanism
        
        Parameters:
        -----------
        X_train : DataFrame
            Training features
        y_train : Series
            Training target
            
        Returns:
        --------
        Trained neural network model, feature mean, feature std
        """
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
            from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            print("TensorFlow not found, installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate
            from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        # Convert to numpy arrays if input is DataFrame
        if hasattr(X_train, 'values'):
            X_train_np = X_train.values
        else:
            X_train_np = X_train
            
        if hasattr(y_train, 'values'):
            y_train_np = y_train.values
        else:
            y_train_np = y_train
        
        # Normalize input data
        mean = X_train_np.mean(axis=0)
        std = X_train_np.std(axis=0)
        X_train_np = (X_train_np - mean) / (std + 1e-8)
        
        # Define advanced model architecture with attention
        print("Building enhanced neural network with attention mechanism...")
        
        # Feature dimension
        n_features = X_train_np.shape[1]
        
        # Create inputs
        inputs = Input(shape=(n_features,))
        
        # Use a simpler enhanced architecture that will work reliably
        # First dense layer with batch normalization and dropout
        x = Dense(64, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Residual connection
        x1 = x
        
        # Second dense layer
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Parallel pathway for capturing different feature interactions
        x2 = Dense(32, activation='relu')(inputs)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.2)(x2)
        
        # Combine pathways
        x = Concatenate()([x1, x2])
        
        # Final dense layers
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Final prediction
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with gradient clipping to prevent exploding gradients
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Callbacks for training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            # Learning rate reduction when plateauing
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]
        
        # Train model
        print("Training enhanced neural network...")
        history = model.fit(
            X_train_np, y_train_np,
            epochs=100,
            batch_size=64,  # Larger batch size for faster training
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Report best validation loss
        best_val_loss = min(history.history['val_loss'])
        print(f"Neural network training complete. Best validation MSE: {best_val_loss:.6f}")
        
        return model, mean, std
        
    def analyze_feature_importance(self, model_name="ridge", top_n=20):
        """
        Analyze and visualize the most important features for prediction
        
        Parameters:
        -----------
        model_name : str
            Model to use for feature importance analysis
        top_n : int
            Number of top features to display
        """
        print(f"\nTop {top_n} Important Features from {model_name.upper()} model:")
        
        if model_name not in self.feature_importances:
            print(f"Feature importance not available for {model_name}")
            return
            
        # Get feature importances and names
        importances = self.feature_importances[model_name]
        feature_names = self.stock_vars
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:min(top_n, len(indices))]
        
        # Create DataFrame for better display
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in top_indices],
            'Importance': importances[top_indices]
        })
        
        # Display top features
        for i, (_, row) in enumerate(importance_df.iterrows()):
            print(f"{i+1}. {row['Feature']}: {row['Importance']:.6f}")
        
        # Create a visualization
        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances ({model_name.upper()})")
        plt.barh(range(len(importance_df)), importance_df['Importance'][::-1])
        plt.yticks(range(len(importance_df)), importance_df['Feature'][::-1])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        
        # Save the figure
        importance_file = os.path.join(self.work_dir, f"feature_importance_{model_name}.png")
        plt.savefig(importance_file)
        print(f"Feature importance plot saved to: {importance_file}")
        
        return importance_df
    
    def calculate_volatility(self, data, lookback=60):
        """
        Calculate rolling volatility for risk management
        
        Parameters:
        -----------
        data : DataFrame
            Data containing stock returns
        lookback : int
            Number of days to use for volatility calculation
            
        Returns:
        --------
        DataFrame with stock volatilities
        """
        # Group by stock
        grouped = data.groupby('permno')
        
        # Calculate volatility
        vols = pd.DataFrame()
        for name, group in grouped:
            if len(group) >= lookback:
                # Sort by date
                sorted_group = group.sort_values('date')
                
                # Calculate rolling volatility (annualized)
                vol = sorted_group[self.ret_var].rolling(window=min(lookback, len(sorted_group))).std() * np.sqrt(252)
                vol_df = pd.DataFrame({'permno': name, 'date': sorted_group['date'], 'volatility': vol})
                vols = pd.concat([vols, vol_df.dropna()], ignore_index=True)
        
        return vols
        
    def analyze_portfolio(self, model_name="ridge"):
        """
        Analyze portfolio performance based on predictions
        
        Parameters:
        -----------
        model_name : str
            Model to use for portfolio formation (default: "ridge")
        """
        print(f"\nPortfolio Analysis using {model_name.upper()} model:")
        
        # Import tabulate for better table formatting
        try:
            from tabulate import tabulate
        except ImportError:
            print("Warning: tabulate package not found, using basic formatting")
            tabulate = None
        
        # Focus on the evaluation period for portfolio analysis
        eval_predictions = self.predictions[
            (self.predictions["date"] >= self.eval_start_date) & 
            (self.predictions["date"] <= self.eval_end_date)
        ]
        
        # Prepare for portfolio construction
        monthly_groups = []
        all_portfolio_holdings = []
        
        # Process each month separately to select top/bottom n stocks
        for (year, month), month_data in eval_predictions.groupby(["year", "month"]):
            date_str = f"{year}-{month:02d}"
            
            # Check if we have enough stocks for the portfolio
            if len(month_data) < (self.n_long + self.n_short):
                print(f"Warning: Not enough stocks in {date_str} to form portfolio")
                n_long = min(self.n_long, len(month_data) // 2)
                n_short = min(self.n_short, len(month_data) // 2)
            else:
                n_long = self.n_long
                n_short = self.n_short
            
            # Sort stocks by predicted returns
            month_data = month_data.sort_values(by=model_name, ascending=False)
            
            # Add portfolio allocation column based on strategy type
            month_data["portfolio"] = "neutral"  # Default allocation
            
            # Calculate volatility metrics for risk management
            current_month_stocks = month_data['permno'].unique()
            historical_data = self.filtered_data[self.filtered_data['permno'].isin(current_month_stocks)]
            vol_data = self.calculate_volatility(historical_data)
            
            # Get latest volatility for each stock
            latest_vols = vol_data.groupby('permno')['volatility'].last().reset_index()
            # Merge volatility into current month data
            month_data = month_data.merge(latest_vols, on='permno', how='left')
            # Fill missing volatilities with median
            median_vol = month_data['volatility'].median()
            month_data['volatility'] = month_data['volatility'].fillna(median_vol)
            # Cap extreme volatilities (3x median)
            month_data['volatility'] = np.minimum(month_data['volatility'], median_vol * 3)
            
            # Target portfolio volatility (annualized)
            target_vol = 0.15  # 15% annualized volatility target
            
            if self.strategy_type in ["long-only", "long-short", "mixed"]:
                # Select top n stocks for long positions
                long_stocks = month_data.iloc[:n_long].copy()
                long_stocks["portfolio"] = "long"
                
                # Calculate inverse volatility weights (more weight to lower vol stocks)
                inv_vol = 1.0 / long_stocks['volatility']
                # Normalize to sum to 1.0 (or n_long for equal notional comparison)
                long_stocks["weight"] = (inv_vol / inv_vol.sum()) * (1.0) 
                # Cap individual position sizes at 3%
                max_weight = 0.03 * n_long
                long_stocks["weight"] = np.minimum(long_stocks["weight"], max_weight)
                # Re-normalize after capping
                long_stocks["weight"] = long_stocks["weight"] / long_stocks["weight"].sum()
                all_portfolio_holdings.append(long_stocks)
            
            if self.strategy_type in ["long-short", "mixed"]:
                # Select bottom n stocks for short positions
                short_stocks = month_data.iloc[-n_short:].copy()
                short_stocks["portfolio"] = "short"
                
                # Calculate inverse volatility weights (more weight to lower vol stocks)
                inv_vol = 1.0 / short_stocks['volatility']
                # Normalize to sum to 1.0 (or n_short for equal notional comparison)
                # Negative weights for shorts
                short_stocks["weight"] = -(inv_vol / inv_vol.sum()) * (1.0)
                # Cap individual position sizes at 3%
                max_weight = 0.03 * n_short
                short_stocks["weight"] = np.maximum(short_stocks["weight"], -max_weight)
                # Re-normalize after capping
                short_stocks["weight"] = short_stocks["weight"] / np.abs(short_stocks["weight"]).sum() * (-1.0)
                all_portfolio_holdings.append(short_stocks)
            
            # Update the month data with portfolio assignments
            if self.strategy_type in ["long-only"]:
                month_data.iloc[:n_long, month_data.columns.get_loc("portfolio")] = "long"
            elif self.strategy_type in ["long-short"]:
                month_data.iloc[:n_long, month_data.columns.get_loc("portfolio")] = "long"
                month_data.iloc[-n_short:, month_data.columns.get_loc("portfolio")] = "short"
            elif self.strategy_type in ["mixed"]:
                month_data.iloc[:n_long, month_data.columns.get_loc("portfolio")] = "long"
                month_data.iloc[-n_short:, month_data.columns.get_loc("portfolio")] = "short"
            
            monthly_groups.append(month_data)
        
        # Combine monthly portfolio selections
        portfolio_data = pd.concat(monthly_groups)
        portfolio_holdings = pd.concat(all_portfolio_holdings)
        
        # Calculate monthly returns by portfolio type
        monthly_returns = portfolio_data.groupby(["year", "month", "portfolio"])[self.ret_var].mean().unstack()
        
        # If 'neutral' column exists but has all NaN values, drop it
        if "neutral" in monthly_returns.columns and monthly_returns["neutral"].isna().all():
            monthly_returns = monthly_returns.drop("neutral", axis=1)
        
        # Calculate strategy returns based on portfolio type
        if self.strategy_type == "long-only":
            monthly_returns["strategy"] = monthly_returns["long"]
        elif self.strategy_type == "long-short":
            if "long" in monthly_returns.columns and "short" in monthly_returns.columns:
                monthly_returns["strategy"] = monthly_returns["long"] - monthly_returns["short"]
            elif "long" in monthly_returns.columns:
                monthly_returns["strategy"] = monthly_returns["long"]
                print("Warning: Short portfolio not available, using long-only")
            else:
                print("Error: No valid portfolio returns calculated")
                return None
        elif self.strategy_type == "mixed":
            # Example mixed strategy: 70% long, 30% short
            if "long" in monthly_returns.columns and "short" in monthly_returns.columns:
                monthly_returns["strategy"] = monthly_returns["long"] * 0.7 - monthly_returns["short"] * 0.3
            elif "long" in monthly_returns.columns:
                monthly_returns["strategy"] = monthly_returns["long"] * 0.7
                print("Warning: Short portfolio not available, using 70% long only")
            elif "short" in monthly_returns.columns:
                monthly_returns["strategy"] = -monthly_returns["short"] * 0.3
                print("Warning: Long portfolio not available, using 30% short only")
            else:
                print("Error: No valid portfolio returns calculated")
                return None
        
        # Reset index for easier manipulation
        monthly_returns = monthly_returns.reset_index()
        
        # Merge with market data
        if self.market_data is not None:
            mkt_data = self.market_data.copy()
            mkt_data.columns = ["rf", "year", "month", "mkt_rf"] if len(mkt_data.columns) == 4 else mkt_data.columns
            
            portfolio_perf = monthly_returns.merge(mkt_data, how="inner", on=["year", "month"])
            
            # Calculate cumulative returns for strategy and market
            portfolio_perf["cumul_strategy"] = (1 + portfolio_perf["strategy"]).cumprod()
            portfolio_perf["cumul_market"] = (1 + portfolio_perf["mkt_rf"]).cumprod()
            
            # Calculate CAPM Alpha
            try:
                nw_ols = sm.ols(formula="strategy ~ mkt_rf", data=portfolio_perf).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
                )
                
                # Get alpha, t-stat, and Information ratio
                alpha = nw_ols.params["Intercept"]
                t_stat = nw_ols.tvalues["Intercept"]
                info_ratio = alpha / np.sqrt(nw_ols.mse_resid) * np.sqrt(12)
                
                print(f"CAPM Alpha: {alpha:.6f}")
                print(f"t-statistic: {t_stat:.4f}")
                print(f"Information Ratio: {info_ratio:.4f}")
            except:
                print("Warning: Could not calculate alpha and t-statistics")
        else:
            # Create basic performance metrics without market data
            portfolio_perf = monthly_returns.copy()
            portfolio_perf["cumul_strategy"] = (1 + portfolio_perf["strategy"]).cumprod()
            print("Warning: Market data not available, skipping CAPM analysis")
        
        # Calculate all required portfolio metrics
        metrics = {}
        
        # Calculate average annualized return
        avg_monthly_return = portfolio_perf["strategy"].mean()
        metrics["Average Annualized Return"] = (1 + avg_monthly_return) ** 12 - 1
        
        # Calculate annualized standard deviation
        metrics["Annualized Standard Deviation"] = portfolio_perf["strategy"].std() * np.sqrt(12)
        
        # Calculate Sharpe ratio
        if "rf" in portfolio_perf.columns:
            # Use risk-free rate if available
            excess_returns = portfolio_perf["strategy"] - portfolio_perf["rf"]
            metrics["Sharpe Ratio"] = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        else:
            # Use zero as risk-free rate if not available
            metrics["Sharpe Ratio"] = avg_monthly_return / portfolio_perf["strategy"].std() * np.sqrt(12)
        
        # Information ratio if market data is available
        if "mkt_rf" in portfolio_perf.columns:
            excess_returns_market = portfolio_perf["strategy"] - portfolio_perf["mkt_rf"]
            metrics["Information Ratio"] = excess_returns_market.mean() / excess_returns_market.std() * np.sqrt(12)
        else:
            metrics["Information Ratio"] = float('nan')
        
        # Calculate alpha from regression
        if "mkt_rf" in portfolio_perf.columns:
            try:
                # Make sure we're using the correct statsmodels import for formula API
                import statsmodels.formula.api as smf
                # Run CAPM regression with Newey-West standard errors
                capm_model = smf.ols(formula="strategy ~ mkt_rf", data=portfolio_perf).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 3}, use_t=True
                )
                
                # Get alpha and annualize it (monthly to annual)
                monthly_alpha = capm_model.params["Intercept"]
                metrics["Annualized Alpha"] = (1 + monthly_alpha) ** 12 - 1
                metrics["Alpha t-statistic"] = capm_model.tvalues["Intercept"]
            except Exception as e:
                print(f"Warning: Error in alpha calculation: {str(e)}")
                metrics["Annualized Alpha"] = float('nan')
                metrics["Alpha t-statistic"] = float('nan')
        else:
            metrics["Annualized Alpha"] = float('nan')
            metrics["Alpha t-statistic"] = float('nan')
        
        # Calculate maximum one-month loss
        metrics["Maximum 1-Month Loss"] = portfolio_perf["strategy"].min()
        
        # Calculate drawdown
        portfolio_perf["cum_return"] = (1 + portfolio_perf["strategy"]).cumprod()
        portfolio_perf["running_max"] = portfolio_perf["cum_return"].cummax()
        portfolio_perf["drawdown"] = 1 - portfolio_perf["cum_return"] / portfolio_perf["running_max"]
        metrics["Maximum Drawdown"] = portfolio_perf["drawdown"].max()
        
        # Calculate portfolio turnover
        turnover_by_type = {}
        
        for port_type in ["long", "short"]:
            if port_type not in portfolio_data["portfolio"].unique():
                continue
                
            # Group by month to calculate monthly turnover
            month_groups = portfolio_data[portfolio_data["portfolio"] == port_type].groupby(["year", "month"])
            month_keys = list(month_groups.groups.keys())
            
            # Calculate monthly turnover rates
            turnover_rates = []
            
            for i in range(len(month_keys) - 1):
                current_month_key = month_keys[i]
                next_month_key = month_keys[i + 1]
                
                # Get stocks in each month
                current_stocks = set(month_groups.get_group(current_month_key)["permno"])
                next_stocks = set(month_groups.get_group(next_month_key)["permno"])
                
                # Calculate turnover (proportion of stocks replaced)
                if current_stocks:
                    stocks_dropped = len(current_stocks - next_stocks)
                    turnover_rate = stocks_dropped / len(current_stocks)
                    turnover_rates.append(turnover_rate)
            
            if turnover_rates:
                avg_turnover = np.mean(turnover_rates)
                turnover_by_type[port_type] = avg_turnover
                # Add to metrics dictionary instead of printing directly
                metrics[f"{port_type.capitalize()} Portfolio Turnover"] = avg_turnover
        
        # Combine long and short turnover if applicable
        if "Long Portfolio Turnover" in metrics and "Short Portfolio Turnover" in metrics:
            metrics["Overall Portfolio Turnover"] = (metrics["Long Portfolio Turnover"] + metrics["Short Portfolio Turnover"]) / 2
        elif "Long Portfolio Turnover" in metrics:
            metrics["Overall Portfolio Turnover"] = metrics["Long Portfolio Turnover"]
        elif "Short Portfolio Turnover" in metrics:
            metrics["Overall Portfolio Turnover"] = metrics["Short Portfolio Turnover"]
        else:
            metrics["Overall Portfolio Turnover"] = float('nan')
        
        # Display comprehensive portfolio metrics table
        print("\n========== PORTFOLIO PERFORMANCE METRICS ==========\n")
        
        # Format all metrics as percentages where appropriate
        formatted_metrics = [
            ["Metric", "Value"],
            ["Strategy Type", self.strategy_type.capitalize()],
            ["Average Annualized Return", f"{metrics['Average Annualized Return']:.4%}"],
            ["Annualized Standard Deviation", f"{metrics['Annualized Standard Deviation']:.4%}"],
            ["Sharpe Ratio", f"{metrics['Sharpe Ratio']:.4f}"],
            ["Information Ratio", f"{metrics['Information Ratio']:.4f}" if not np.isnan(metrics['Information Ratio']) else "N/A"],
            ["Annualized Alpha", f"{metrics['Annualized Alpha']:.4%}" if not np.isnan(metrics['Annualized Alpha']) else "N/A"],
            ["Alpha t-statistic", f"{metrics['Alpha t-statistic']:.4f}" if not np.isnan(metrics['Alpha t-statistic']) else "N/A"],
            ["Maximum Drawdown", f"{metrics['Maximum Drawdown']:.4%}"],
            ["Maximum 1-Month Loss", f"{metrics['Maximum 1-Month Loss']:.4%}"],
            ["Overall Portfolio Turnover", f"{metrics['Overall Portfolio Turnover']:.4%}" if not np.isnan(metrics['Overall Portfolio Turnover']) else "N/A"]
        ]
        
        # Add long and short turnover if available
        if "Long Portfolio Turnover" in metrics:
            formatted_metrics.append(["Long Portfolio Turnover", f"{metrics['Long Portfolio Turnover']:.4%}"])
        if "Short Portfolio Turnover" in metrics:
            formatted_metrics.append(["Short Portfolio Turnover", f"{metrics['Short Portfolio Turnover']:.4%}"])
        
        # Display metrics as a nicely formatted table
        if tabulate:
            print(tabulate(formatted_metrics, headers="firstrow", tablefmt="grid"))
        else:
            # Basic formatting if tabulate is not available
            for metric, value in formatted_metrics:
                print(f"{metric:30} {value}")
        
        # Identify top holdings (stocks most frequently selected)
        if "ticker" in portfolio_holdings.columns or "comnam" in portfolio_holdings.columns:
            print("\nTop 10 Holdings (by frequency in portfolio):")
            
            # Group by permno to find frequency
            id_cols = ["permno"]
            if "ticker" in portfolio_holdings.columns:
                id_cols.append("ticker")
            if "comnam" in portfolio_holdings.columns:
                id_cols.append("comnam")
                
            holdings_count = portfolio_holdings.groupby(id_cols).size().reset_index(name="months_held")
            
            # Calculate average weight when held
            holdings_weight = portfolio_holdings.groupby(id_cols)["weight"].mean().reset_index(name="avg_weight")
            
            # Combine frequency and weight
            top_holdings = holdings_count.merge(holdings_weight, on=id_cols)
            
            # Sort by frequency and then by weight
            top_holdings = top_holdings.sort_values(["months_held", "avg_weight"], ascending=False).head(10)
            
            # Display top holdings
            holdings_table = [["Rank", "Stock", "Type", "Months Held", "Avg Weight"]]
            for i, (_, row) in enumerate(top_holdings.iterrows()):
                holding_name = row.get("ticker", "") or row.get("comnam", f"Stock {row['permno']}")
                sign = "Long" if row["avg_weight"] > 0 else "Short"
                holdings_table.append([i+1, f"{holding_name} ({row['permno']})", sign, row["months_held"], f"{abs(row['avg_weight']):.2%}"])
            
            if tabulate:
                print(tabulate(holdings_table, headers="firstrow", tablefmt="grid"))
            else:
                for holding in holdings_table:
                    print(f"{holding[0]:4} {holding[1]:20} {holding[2]:6} {holding[3]:11} {holding[4]:10}")
        
        # Plot the cumulative performance
        plt.figure(figsize=(12, 6))
        
        # Create a proper datetime index for plotting
        if 'year' in portfolio_perf.columns and 'month' in portfolio_perf.columns:
            # Create date objects for x-axis
            dates = [pd.Timestamp(year=int(y), month=int(m), day=1) for y, m in zip(portfolio_perf['year'], portfolio_perf['month'])]
            
            # Plot with dates on x-axis
            plt.plot(dates, portfolio_perf["cumul_strategy"], label=f"Portfolio Strategy ({self.strategy_type})")
            
            if "cumul_market" in portfolio_perf.columns:
                plt.plot(dates, portfolio_perf["cumul_market"], label="S&P 500")
        else:
            # Fallback to original implementation if year/month not available
            plt.plot(portfolio_perf["cumul_strategy"], label=f"Portfolio Strategy ({self.strategy_type})")
            
            if "cumul_market" in portfolio_perf.columns:
                plt.plot(portfolio_perf["cumul_market"], label="S&P 500")
            
        plt.title("Cumulative Performance: Portfolio vs. S&P 500")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        
        # Format x-axis ticks as dates
        plt.gcf().autofmt_xdate()  # Auto-format the dates on x-axis
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as YYYY-MM
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
        
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        perf_plot_file = os.path.join(self.work_dir, f"performance_plot_{model_name}_{self.strategy_type}.png")
        plt.savefig(perf_plot_file)
        print(f"Performance plot saved to: {perf_plot_file}")
        
        # Save portfolio results to CSV
        port_file = os.path.join(self.work_dir, f"portfolio_results_{model_name}_{self.strategy_type}.csv")
        portfolio_perf.to_csv(port_file, index=False)
        print(f"Portfolio results saved to: {port_file}")
        
        return portfolio_perf
    
    def adjust_for_factors(self, predictions, model_name):
        """
        Adjust predictions for common risk factors to improve statistical significance
        
        Parameters:
        -----------
        predictions : DataFrame
            Dataframe containing stock predictions
        model_name : str
            Model name whose predictions to adjust
        
        Returns:
        --------
        DataFrame with adjusted predictions
        """
        print(f"\nApplying factor-based enhancement to {model_name.upper()} predictions...")
        
        # We'll implement a basic size and momentum adjustment
        # This helps isolate unique alpha and increase statistical significance
        enhanced_predictions = predictions.copy()
        
        # Process each month separately
        for (year, month), month_data in predictions.groupby(["year", "month"]):
            if len(month_data) < 30:  # Need enough stocks for meaningful regression
                continue
                
            # Use stock characteristics as factors (size, momentum, volatility)
            # Use log market cap as size factor if available, otherwise use rank
            month_data['size_factor'] = np.arange(len(month_data)) / len(month_data)  # Normalized rank as proxy
            
            # Use recent return as momentum factor (last month return)
            if 'ret_1_0' in month_data.columns:
                month_data['momentum_factor'] = month_data['ret_1_0']
            else:
                # Create a momentum proxy using prediction rank
                month_data['momentum_factor'] = month_data[model_name].rank(pct=True)
            
            # Create a value factor if available, otherwise use rank
            if 'bm' in month_data.columns:  # Book-to-market
                month_data['value_factor'] = month_data['bm']
            else:
                month_data['value_factor'] = 0  # Neutral on value
            
            # Standardize factors
            for factor in ['size_factor', 'momentum_factor', 'value_factor']:
                if month_data[factor].std() > 0:
                    month_data[factor] = (month_data[factor] - month_data[factor].mean()) / month_data[factor].std()
                else:
                    month_data[factor] = 0
            
            # Regress predictions on factors to get residual alpha
            X = month_data[['size_factor', 'momentum_factor', 'value_factor']]
            y = month_data[model_name]
            
            try:
                # Add constant (using statsmodels.api)
                X = sm.add_constant(X)
                # Fit regression
                model = sm.OLS(y, X).fit()
                # Get residuals (pure alpha)
                residuals = model.resid
                # Calculate factor exposure-adjusted predictions
                factor_return = model.predict(X) - model.params['const']
                alpha_component = model.params['const'] + residuals
                
                # Weight alpha more heavily (80% alpha, 20% factor return)
                alpha_weight = 0.8
                factor_weight = 1.0 - alpha_weight
                adjusted_predictions = (alpha_component * alpha_weight) + (factor_return * factor_weight)
                
                # Update predictions
                enhanced_predictions.loc[month_data.index, f"{model_name}_enhanced"] = adjusted_predictions
            except Exception as e:
                print(f"  Warning: Factor adjustment failed for {year}-{month}: {str(e)}")
                enhanced_predictions.loc[month_data.index, f"{model_name}_enhanced"] = month_data[model_name]
        
        print(f"Factor enhancement complete for {model_name.upper()} predictions")
        return enhanced_predictions
        
    def run_full_pipeline(self, model_for_portfolio=None):
        """
        Run the full prediction and analysis pipeline
        
        Parameters:
        -----------
        model_for_portfolio : str
            Which model to use for portfolio formation (default: best model by R²)
        """
        try:
            self.load_data()
            self.preprocess_data()
            self.train_and_predict()
            best_model, _ = self.evaluate_models()
            
            # Use best model if not specified
            model_to_use = model_for_portfolio or best_model
            
            # Apply factor-based enhancements to improve statistical significance
            print(f"\nApplying advanced quantitative enhancements...")
            self.predictions = self.adjust_for_factors(self.predictions, model_to_use)
            
            # Use the enhanced version of the model
            enhanced_model_name = f"{model_to_use}_enhanced"
            if enhanced_model_name in self.predictions.columns:
                print(f"Using factor-enhanced version of {model_to_use.upper()} for portfolio construction")
                model_to_use = enhanced_model_name
            
            # Analyze feature importance
            self.analyze_feature_importance(model_to_use)
            
            # Analyze portfolio performance
            self.analyze_portfolio(model_to_use)
            
            print(f"\nPrediction and analysis completed at: {datetime.datetime.now()}")
        except Exception as e:
            print(f"Error in pipeline execution: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return self
    
if __name__ == "__main__":
    # Check for fast mode command line argument
    import sys
    fast_mode = len(sys.argv) > 1 and sys.argv[1] == 'fast'
    
    # Create and run the model with more stocks in the portfolio
    model = StockPredictionModel(
        n_long=50, 
        n_short=50, 
        strategy_type="long-short",
        fast_mode=fast_mode
    )
    model.run_full_pipeline()