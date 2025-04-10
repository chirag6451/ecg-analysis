import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
import shap
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ECGMultiClassifier:
    """
    Advanced multi-classifier approach for ECG arrhythmia detection.
    
    This class implements multiple classifiers (traditional ML and deep learning)
    and combines them for improved accuracy.
    """
    
    def __init__(self, model_type='ensemble', num_classes=2, model_dir='models'):
        """
        Initialize the multi-classifier.
        
        Args:
            model_type (str): Type of model to use:
                - 'ensemble': Ensemble of traditional ML models
                - 'cnn': 1D CNN for raw ECG classification
                - 'hybrid': Combination of ensemble and deep learning
                - 'deep_ensemble': Ensemble of deep learning models
            num_classes (int): Number of classes to predict
            model_dir (str): Directory to save trained models
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.model_dir = model_dir
        self.is_fitted = False
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models based on type
        if model_type in ['ensemble', 'hybrid']:
            self._initialize_ensemble_models()
            
        if model_type in ['cnn', 'hybrid', 'deep_ensemble']:
            self._initialize_deep_learning_models()
            
    def _initialize_ensemble_models(self):
        """Initialize traditional machine learning ensemble models."""
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Neural Network
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42
        )
        
        # Support Vector Machine
        svm = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Store individual models
        self.models['rf'] = rf
        self.models['gb'] = gb
        self.models['xgb'] = xgb_model
        self.models['lgb'] = lgb_model
        self.models['mlp'] = mlp
        self.models['svm'] = svm
        
        # Create voting ensemble
        self.models['voting'] = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('mlp', mlp),
                ('svm', svm)
            ],
            voting='soft'
        )
    
    def _initialize_deep_learning_models(self):
        """Initialize deep learning models."""
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        # Will be initialized during training when input shape is known
        self.models['cnn'] = None
        self.models['lstm'] = None
        self.models['hybrid_net'] = None
        
    def _build_cnn_model(self, input_shape):
        """
        Build a 1D CNN model for raw ECG classification.
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        model = Sequential([
            Conv1D(64, kernel_size=5, activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(128, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Conv1D(256, kernel_size=5, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_hybrid_model(self, signal_shape, feature_shape):
        """
        Build a hybrid model that combines raw signal and extracted features.
        
        Args:
            signal_shape (tuple): Shape of raw signal input
            feature_shape (int): Number of extracted features
            
        Returns:
            tf.keras.Model: Compiled hybrid model
        """
        # Signal branch (CNN)
        signal_input = Input(shape=signal_shape, name='signal_input')
        x1 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(signal_input)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)
        
        x1 = Conv1D(128, kernel_size=5, activation='relu', padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling1D(pool_size=2)(x1)
        
        x1 = Flatten()(x1)
        x1 = Dense(128, activation='relu')(x1)
        x1 = Dropout(0.5)(x1)
        
        # Feature branch (MLP)
        feature_input = Input(shape=(feature_shape,), name='feature_input')
        x2 = Dense(64, activation='relu')(feature_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Combine branches
        combined = tf.keras.layers.concatenate([x1, x2])
        x = Dense(128, activation='relu')(combined)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=[signal_input, feature_input], outputs=output)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fit(self, X, y, X_raw=None, validation_data=None, epochs=50, batch_size=32):
        """
        Train the multi-classifier model.
        
        Args:
            X (np.array or pd.DataFrame): Feature matrix for traditional models
            y (np.array or pd.Series): Target labels
            X_raw (np.array, optional): Raw ECG signals for deep learning models
            validation_data (tuple, optional): Validation data as (X_val, y_val) or 
                                               (X_val, X_raw_val, y_val)
            epochs (int): Number of epochs for deep learning models
            batch_size (int): Batch size for deep learning models
            
        Returns:
            self: Returns an instance of self
        """
        # Save feature names if DataFrame is provided
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert labels to appropriate format
        if self.num_classes > 2:
            # Multi-class
            y_cat = to_categorical(y, num_classes=self.num_classes) if self.model_type in ['cnn', 'hybrid', 'deep_ensemble'] else y
        else:
            # Binary classification
            y_cat = to_categorical(y, num_classes=2) if self.model_type in ['cnn', 'hybrid', 'deep_ensemble'] else y
            
        # Train ensemble models
        if self.model_type in ['ensemble', 'hybrid']:
            print("Training ensemble models...")
            for name, model in self.models.items():
                if name not in ['cnn', 'lstm', 'hybrid_net', 'voting']:
                    print(f"Training {name}...")
                    model.fit(X_scaled, y)
                    
            # Train voting ensemble
            print("Training voting ensemble...")
            self.models['voting'].fit(X_scaled, y)
            
        # Train deep learning models
        if self.model_type in ['cnn', 'hybrid', 'deep_ensemble'] and X_raw is not None:
            print("Training deep learning models...")
            
            # Prepare validation data
            val_data = None
            if validation_data is not None:
                if len(validation_data) == 2:
                    X_val, y_val = validation_data
                    X_val_scaled = self.scaler.transform(X_val)
                    y_val_cat = to_categorical(y_val, num_classes=self.num_classes) if self.num_classes > 1 else y_val
                    val_data = (X_val_scaled, y_val_cat)
                elif len(validation_data) == 3:
                    X_val, X_raw_val, y_val = validation_data
                    X_val_scaled = self.scaler.transform(X_val)
                    y_val_cat = to_categorical(y_val, num_classes=self.num_classes) if self.num_classes > 1 else y_val
                    val_data = (X_raw_val, y_val_cat)
            
            # CNN model (raw signal)
            if X_raw is not None:
                if len(X_raw.shape) == 2:
                    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                
                # Build and train CNN model
                self.models['cnn'] = self._build_cnn_model((X_raw.shape[1], X_raw.shape[2]))
                history = self.models['cnn'].fit(
                    X_raw, y_cat,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=val_data if val_data is not None else None,
                    verbose=1
                )
                
                # Save training history
                self.cnn_history = history.history
                
            # Hybrid model (both raw signal and features)
            if self.model_type == 'hybrid' and X_raw is not None:
                signal_shape = (X_raw.shape[1], X_raw.shape[2])
                feature_shape = X.shape[1]
                
                self.models['hybrid_net'] = self._build_hybrid_model(signal_shape, feature_shape)
                
                # Prepare validation data for hybrid model
                hybrid_val_data = None
                if validation_data is not None and len(validation_data) == 3:
                    X_val, X_raw_val, y_val = validation_data
                    X_val_scaled = self.scaler.transform(X_val)
                    y_val_cat = to_categorical(y_val, num_classes=self.num_classes) if self.num_classes > 1 else y_val
                    
                    # Reshape raw signals if needed
                    if len(X_raw_val.shape) == 2:
                        X_raw_val = X_raw_val.reshape(X_raw_val.shape[0], X_raw_val.shape[1], 1)
                        
                    hybrid_val_data = ({'signal_input': X_raw_val, 'feature_input': X_val_scaled}, y_val_cat)
                
                # Train hybrid model
                history = self.models['hybrid_net'].fit(
                    {'signal_input': X_raw, 'feature_input': X_scaled},
                    y_cat,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=hybrid_val_data,
                    verbose=1
                )
                
                # Save training history
                self.hybrid_history = history.history
        
        self.is_fitted = True
        return self
    
    def predict(self, X, X_raw=None):
        """
        Predict class labels.
        
        Args:
            X (np.array or pd.DataFrame): Feature matrix
            X_raw (np.array, optional): Raw ECG signals
            
        Returns:
            np.array: Predicted class labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        
        # Predict based on model type
        if self.model_type == 'ensemble':
            # Use voting classifier for predictions
            return self.models['voting'].predict(X_scaled)
            
        elif self.model_type == 'cnn':
            if X_raw is None:
                raise ValueError("Raw ECG signal required for CNN prediction")
                
            # Prepare raw signals
            if len(X_raw.shape) == 2:
                X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                
            # Get predictions from CNN
            probs = self.models['cnn'].predict(X_raw)
            return np.argmax(probs, axis=1)
            
        elif self.model_type == 'hybrid':
            # Combine predictions from ensemble and deep learning
            # Use voting classifier for ensemble part
            ensemble_pred = self.models['voting'].predict_proba(X_scaled)
            
            # Use hybrid model for deep learning part
            if X_raw is not None:
                if len(X_raw.shape) == 2:
                    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                    
                hybrid_pred = self.models['hybrid_net'].predict({
                    'signal_input': X_raw,
                    'feature_input': X_scaled
                })
                
                # Average the probabilities
                final_pred = (ensemble_pred + hybrid_pred) / 2
                return np.argmax(final_pred, axis=1)
            else:
                # Fall back to just ensemble if raw signal not provided
                return np.argmax(ensemble_pred, axis=1)
                
        elif self.model_type == 'deep_ensemble':
            if X_raw is None:
                raise ValueError("Raw ECG signal required for deep ensemble prediction")
                
            # Prepare raw signals
            if len(X_raw.shape) == 2:
                X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                
            # Use CNN predictions
            probs = self.models['cnn'].predict(X_raw)
            return np.argmax(probs, axis=1)
        
    def predict_proba(self, X, X_raw=None):
        """
        Predict class probabilities.
        
        Args:
            X (np.array or pd.DataFrame): Feature matrix
            X_raw (np.array, optional): Raw ECG signals
            
        Returns:
            np.array: Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        # Prepare features
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        X_scaled = self.scaler.transform(X)
        
        # Predict based on model type
        if self.model_type == 'ensemble':
            # Use voting classifier for probability predictions
            return self.models['voting'].predict_proba(X_scaled)
            
        elif self.model_type == 'cnn':
            if X_raw is None:
                raise ValueError("Raw ECG signal required for CNN prediction")
                
            # Prepare raw signals
            if len(X_raw.shape) == 2:
                X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                
            # Get probabilities from CNN
            return self.models['cnn'].predict(X_raw)
            
        elif self.model_type == 'hybrid':
            # Combine predictions from ensemble and deep learning
            # Use voting classifier for ensemble part
            ensemble_pred = self.models['voting'].predict_proba(X_scaled)
            
            # Use hybrid model for deep learning part
            if X_raw is not None:
                if len(X_raw.shape) == 2:
                    X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                    
                hybrid_pred = self.models['hybrid_net'].predict({
                    'signal_input': X_raw,
                    'feature_input': X_scaled
                })
                
                # Average the probabilities
                final_pred = (ensemble_pred + hybrid_pred) / 2
                return final_pred
            else:
                # Fall back to just ensemble if raw signal not provided
                return ensemble_pred
                
        elif self.model_type == 'deep_ensemble':
            if X_raw is None:
                raise ValueError("Raw ECG signal required for deep ensemble prediction")
                
            # Prepare raw signals
            if len(X_raw.shape) == 2:
                X_raw = X_raw.reshape(X_raw.shape[0], X_raw.shape[1], 1)
                
            # Use CNN predictions
            return self.models['cnn'].predict(X_raw)
    
    def save_models(self):
        """Save all trained models to disk."""
        if not self.is_fitted:
            raise ValueError("Models not fitted yet. Call 'fit' first.")
            
        # Save feature names
        if self.feature_names is not None:
            np.save(os.path.join(self.model_dir, 'feature_names.npy'), np.array(self.feature_names))
            
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Save traditional ML models
        if self.model_type in ['ensemble', 'hybrid']:
            for name, model in self.models.items():
                if name not in ['cnn', 'lstm', 'hybrid_net']:
                    joblib.dump(model, os.path.join(self.model_dir, f'{name}_model.pkl'))
        
        # Save deep learning models
        if self.model_type in ['cnn', 'hybrid', 'deep_ensemble']:
            if self.models['cnn'] is not None:
                self.models['cnn'].save(os.path.join(self.model_dir, 'cnn_model'))
                
            if 'hybrid_net' in self.models and self.models['hybrid_net'] is not None:
                self.models['hybrid_net'].save(os.path.join(self.model_dir, 'hybrid_model'))
                
        print(f"Models saved to {self.model_dir}")
        
    def load_models(self):
        """Load trained models from disk."""
        # Load feature names
        feature_names_path = os.path.join(self.model_dir, 'feature_names.npy')
        if os.path.exists(feature_names_path):
            self.feature_names = np.load(feature_names_path, allow_pickle=True).tolist()
            
        # Load scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        # Load traditional ML models
        if self.model_type in ['ensemble', 'hybrid']:
            for name in ['rf', 'gb', 'xgb', 'lgb', 'mlp', 'svm', 'voting']:
                model_path = os.path.join(self.model_dir, f'{name}_model.pkl')
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    
        # Load deep learning models
        if self.model_type in ['cnn', 'hybrid', 'deep_ensemble']:
            cnn_path = os.path.join(self.model_dir, 'cnn_model')
            if os.path.exists(cnn_path):
                self.models['cnn'] = tf.keras.models.load_model(cnn_path)
                
            hybrid_path = os.path.join(self.model_dir, 'hybrid_model')
            if os.path.exists(hybrid_path):
                self.models['hybrid_net'] = tf.keras.models.load_model(hybrid_path)
                
        self.is_fitted = True
        print(f"Models loaded from {self.model_dir}")
        
    def get_feature_importance(self, X, y=None, plot=False):
        """
        Get feature importance for ensemble models.
        
        Args:
            X (np.array or pd.DataFrame): Feature matrix
            y (np.array, optional): Target labels (for SHAP)
            plot (bool): Whether to plot feature importance
            
        Returns:
            pd.DataFrame: Feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        if self.model_type not in ['ensemble', 'hybrid']:
            raise ValueError("Feature importance only available for ensemble models")
            
        # Prepare features
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            feature_names = X.columns.tolist()
        else:
            X_values = X
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
        X_scaled = self.scaler.transform(X_values)
        
        # Get feature importance from random forest
        if 'rf' in self.models:
            rf_importance = self.models['rf'].feature_importances_
            rf_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_importance
            }).sort_values('Importance', ascending=False)
            
            if plot:
                plt.figure(figsize=(10, 6))
                plt.barh(rf_importance_df['Feature'][:20], rf_importance_df['Importance'][:20])
                plt.xlabel('Importance')
                plt.title('Random Forest Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()
                
            return rf_importance_df
            
    def get_shap_values(self, X, model_name='rf', plot=True):
        """
        Calculate SHAP values for model explainability.
        
        Args:
            X (np.array or pd.DataFrame): Feature matrix
            model_name (str): Name of the model to explain
            plot (bool): Whether to plot SHAP values
            
        Returns:
            np.array: SHAP values
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
            
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
            
        # Prepare features
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            feature_names = X.columns.tolist()
        else:
            X_values = X
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
        X_scaled = self.scaler.transform(X_values)
        
        # Get model
        model = self.models[model_name]
        
        # Create explainer
        if model_name in ['rf', 'gb', 'xgb', 'lgb']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            
            if plot:
                if isinstance(shap_values, list):  # For multiclass
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values[1] if len(shap_values) > 1 else shap_values[0], 
                                     X_scaled, feature_names=feature_names)
                else:
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names)
                
            return shap_values
        else:
            print(f"SHAP values not implemented for model type '{model_name}'")
            return None 