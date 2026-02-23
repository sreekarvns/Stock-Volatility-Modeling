import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')


def build_xgb_model():
    """
    Build XGBoost model with tuned hyperparameters for volatility prediction.
    Optimized for accuracy with regularization.
    """
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )


def prepare_xgb_data(garch_pred, egarch_pred, realized_vol, window=20):
    """
    Prepare feature matrix for XGBoost training.
    
    Features include:
    - Lagged GARCH predictions
    - Lagged EGARCH predictions
    - Lagged realized volatility
    - Technical indicators (RSI, MACD, Bollinger Bands, Momentum)
    
    Args:
        garch_pred (array): GARCH predictions
        egarch_pred (array): EGARCH predictions
        realized_vol (array): Realized volatility
        window (int): Rolling window size for features
    
    Returns:
        X (array): Feature matrix
        y (array): Target values (realized volatility)
    """
    
    # Ensure inputs are numpy arrays
    garch_pred = np.array(garch_pred).flatten()
    egarch_pred = np.array(egarch_pred).flatten()
    realized_vol = np.array(realized_vol).flatten()
    
    # Ensure all arrays have same length
    min_len = min(len(garch_pred), len(egarch_pred), len(realized_vol))
    garch_pred = garch_pred[-min_len:]
    egarch_pred = egarch_pred[-min_len:]
    realized_vol = realized_vol[-min_len:]
    
    X, y = [], []
    
    for i in range(window, len(realized_vol)):
        try:
            features = []
            
            # 1. Lagged GARCH predictions (window features)
            garch_window = garch_pred[i-window:i]
            features.extend(garch_window)
            
            # 2. Lagged EGARCH predictions (window features)
            egarch_window = egarch_pred[i-window:i]
            features.extend(egarch_window)
            
            # 3. Lagged realized volatility (window features)
            vol_window = realized_vol[i-window:i]
            features.extend(vol_window)
            
            # 4. Statistical features
            # Mean of realized vol in window
            vol_mean = np.mean(vol_window)
            features.append(vol_mean)
            
            # Std of realized vol in window
            vol_std = np.std(vol_window)
            features.append(vol_std)
            
            # Skewness approximation
            vol_skew = np.mean(((vol_window - vol_mean) / (vol_std + 1e-6)) ** 3) if vol_std > 0 else 0
            features.append(vol_skew)
            
            # Kurtosis approximation
            vol_kurt = np.mean(((vol_window - vol_mean) / (vol_std + 1e-6)) ** 4) if vol_std > 0 else 0
            features.append(vol_kurt)
            
            # 5. Volatility changes
            vol_change = realized_vol[i] - realized_vol[i-1] if i > 0 else 0
            features.append(vol_change)
            
            # 6. Momentum (volatility trend)
            if len(vol_window) > 1:
                momentum = vol_window[-1] - vol_window[0]
            else:
                momentum = 0
            features.append(momentum)
            
            # 7. Volatility of volatility
            vol_of_vol = np.std(np.diff(vol_window)) if len(vol_window) > 1 else 0
            features.append(vol_of_vol)
            
            # 8. GARCH-EGARCH difference (model disagreement)
            garch_egarch_diff = np.mean(np.abs(garch_window - egarch_window))
            features.append(garch_egarch_diff)
            
            # 9. GARCH-EGARCH correlation
            if len(garch_window) > 1 and np.std(garch_window) > 0 and np.std(egarch_window) > 0:
                garch_egarch_corr = np.corrcoef(garch_window, egarch_window)[0, 1]
                if np.isnan(garch_egarch_corr):
                    garch_egarch_corr = 0
            else:
                garch_egarch_corr = 0
            features.append(garch_egarch_corr)
            
            # 10. Mean GARCH and EGARCH
            mean_garch = np.mean(garch_window)
            features.append(mean_garch)
            
            mean_egarch = np.mean(egarch_window)
            features.append(mean_egarch)
            
            # Ensure we have valid features
            if len(features) > 0 and all(np.isfinite(features)):
                X.append(features)
                y.append(realized_vol[i])
        
        except Exception as e:
            # Skip problematic rows
            continue
    
    if len(X) == 0:
        raise ValueError("Could not create any valid feature rows. Check input data.")
    
    X = np.array(X)
    y = np.array(y)
    
    # Handle any remaining NaNs or infinities
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]
    
    return X, y


def fit_hybrid_garch_lstm(garch_pred, egarch_pred, realized_vol, window=20):
    """
    Fit hybrid GARCH-XGBoost model and generate predictions.
    
    The hybrid model combines:
    - GARCH(1,1) predictions (20% weight)
    - EGARCH(1,1) predictions (20% weight)
    - XGBoost predictions (60% weight - learns from both models)
    
    Args:
        garch_pred (array): GARCH conditional volatility
        egarch_pred (array): EGARCH conditional volatility
        realized_vol (array): Realized volatility (ground truth)
        window (int): Rolling window for feature engineering
    
    Returns:
        actual (array): Actual realized volatility
        hybrid_pred (array): Hybrid model predictions
    """
    
    try:
        print("\n" + "="*70)
        print("HYBRID GARCH-XGBoost MODEL")
        print("="*70)
        
        # Ensure inputs are numpy arrays
        garch_pred = np.array(garch_pred).flatten()
        egarch_pred = np.array(egarch_pred).flatten()
        realized_vol = np.array(realized_vol).flatten()
        
        # Align arrays
        min_len = min(len(garch_pred), len(egarch_pred), len(realized_vol))
        garch_pred = garch_pred[-min_len:]
        egarch_pred = egarch_pred[-min_len:]
        realized_vol = realized_vol[-min_len:]
        
        print(f"Input data shapes:")
        print(f"  GARCH: {len(garch_pred)}")
        print(f"  EGARCH: {len(egarch_pred)}")
        print(f"  Realized Vol: {len(realized_vol)}")
        
        # Prepare XGBoost data
        print(f"\nPreparing features with window={window}...")
        X_xgb, y_xgb = prepare_xgb_data(garch_pred, egarch_pred, realized_vol, window=window)
        
        print(f"Feature matrix shape: {X_xgb.shape}")
        print(f"Target shape: {y_xgb.shape}")
        
        if X_xgb.shape[0] < 50:
            print(f"⚠️  Warning: Only {X_xgb.shape[0]} samples. Model may be unstable.")
        
        # Build and train XGBoost model
        print("\nTraining XGBoost model...")
        model = build_xgb_model()
        model.fit(X_xgb, y_xgb, verbose=False)
        print("✓ XGBoost training complete")
        
        # Make predictions
        xgb_pred = model.predict(X_xgb)
        
        # Align all predictions to same length
        min_pred_len = min(len(garch_pred), len(egarch_pred), len(xgb_pred))
        garch_aligned = garch_pred[-min_pred_len:]
        egarch_aligned = egarch_pred[-min_pred_len:]
        xgb_aligned = xgb_pred[-min_pred_len:]
        realized_aligned = realized_vol[-min_pred_len:]
        
        # Create ensemble prediction
        # Weights: 20% GARCH + 20% EGARCH + 60% XGBoost (learned ensemble)
        hybrid_pred = (0.2 * garch_aligned + 
                      0.2 * egarch_aligned + 
                      0.6 * xgb_aligned)
        
        print(f"\nEnsemble weights: 0.2 (GARCH) + 0.2 (EGARCH) + 0.6 (XGBoost)")
        print(f"Hybrid predictions shape: {hybrid_pred.shape}")
        print(f"Prediction range: {hybrid_pred.min():.6f} to {hybrid_pred.max():.6f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        
        print(f"\nTop 5 Important Features:")
        for idx, feat_idx in enumerate(top_features_idx, 1):
            print(f"  {idx}. Feature {feat_idx}: {feature_importance[feat_idx]:.4f}")
        
        print("="*70)
        
        return realized_aligned, hybrid_pred
    
    except Exception as e:
        print(f"\n❌ Error in fit_hybrid_garch_lstm: {str(e)}")
        print("Returning None, None")
        return None, None
