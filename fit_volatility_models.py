from arch import arch_model
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def fit_volatility_models(log_returns):
    """
    Fits ARCH(1), GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1) models.
    
    ⭐ CRITICAL: Scales returns by 100 before fitting
    
    Args:
        log_returns (pd.Series): Series of log-returns
    Returns:
        dict: Dictionary with fitted model objects
    """
    if not isinstance(log_returns, pd.Series):
        raise ValueError("Input must be a pandas Series.")
    
    # ⭐ THE ONE CRITICAL LINE
    clean_returns = log_returns * 100
    print(f"✓ Returns scaled: mean={clean_returns.mean():.6f}, std={clean_returns.std():.6f}")
    
    # Remove outliers (beyond 5 stddev)
    mean = clean_returns.mean()
    std = clean_returns.std()
    clean_returns = clean_returns[np.abs(clean_returns - mean) <= 5 * std].dropna()
    print(f"✓ After outlier removal: {len(clean_returns)} observations\n")
    
    models = {}
    
    try:
        # ARCH(1)
        print("Fitting ARCH(1)...")
        arch1 = arch_model(clean_returns, vol='ARCH', p=1, o=0, q=0, dist='t').fit(
            disp='off', show_warning=False
        )
        print(f"✓ ARCH(1) - AIC: {arch1.aic:.2f}, BIC: {arch1.bic:.2f}")
        models['ARCH(1)'] = arch1
        
        # GARCH(1,1)
        print("Fitting GARCH(1,1)...")
        garch11 = arch_model(clean_returns, vol='GARCH', p=1, o=0, q=1, dist='t').fit(
            disp='off', show_warning=False
        )
        print(f"✓ GARCH(1,1) - AIC: {garch11.aic:.2f}, BIC: {garch11.bic:.2f}")
        models['GARCH(1,1)'] = garch11
        
        # GJR-GARCH(1,1)
        print("Fitting GJR-GARCH(1,1)...")
        try:
            gjr_garch = arch_model(clean_returns, vol='GARCH', p=1, o=1, q=1, dist='t').fit(
                disp='off', show_warning=False
            )
            print(f"✓ GJR-GARCH(1,1) - AIC: {gjr_garch.aic:.2f}, BIC: {gjr_garch.bic:.2f}")
            models['GJR-GARCH(1,1)'] = gjr_garch
        except Exception as e:
            print(f"⚠️  GJR-GARCH failed (this is OK): {str(e)[:50]}")
        
        # EGARCH(1,1)
        print("Fitting EGARCH(1,1)...")
        egarch = arch_model(clean_returns, vol='EGARCH', p=1, o=0, q=1, dist='t').fit(
            disp='off', show_warning=False
        )
        print(f"✓ EGARCH(1,1) - AIC: {egarch.aic:.2f}, BIC: {egarch.bic:.2f}")
        models['EGARCH(1,1)'] = egarch

        # --- Fit Hybrid GARCH-LSTM ---
        try:
            from hybrid_garch_lstm import fit_hybrid_garch_lstm
            # Prepare realized volatility (rolling std)
            realized_vol = clean_returns.rolling(window=20).std().dropna()
            # Align predictions
            garch_pred = np.sqrt(garch11.conditional_volatility[-len(realized_vol):])
            egarch_pred = np.sqrt(egarch.conditional_volatility[-len(realized_vol):])
            realized_pred = realized_vol[-len(garch_pred):]
            actual, hybrid_pred = fit_hybrid_garch_lstm(
                garch_pred,
                egarch_pred,
                realized_pred,
                window=20
            )
            if actual is not None and hybrid_pred is not None:
                models['Hybrid GARCH-LSTM'] = {
                    'actual': actual,
                    'predicted': hybrid_pred
                }
                print("✓ Hybrid GARCH-LSTM model fitted and added to models dict.")
            else:
                print("❌ Hybrid GARCH-LSTM model fitting failed.")
        except Exception as e:
            print(f"❌ Error fitting Hybrid GARCH-LSTM: {str(e)}")

    except Exception as e:
        print(f"Error fitting models: {e}")
    
    return models