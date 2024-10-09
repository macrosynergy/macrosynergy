import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from itertools import product

def optimize_arima(data, p_range, d_range, q_range, criterion='AIC'):
    """
    Perform grid search to optimize ARIMA parameters based on AIC or BIC.
    
    Parameters:
    - data: Time series data
    - p_range: Range of values for the AR term (p)
    - d_range: Range of values for the differencing term (d)
    - q_range: Range of values for the MA term (q)
    - criterion: 'AIC' or 'BIC' for model selection
    
    Returns:
    - Best parameters and the corresponding AIC/BIC value
    """
    best_score = np.inf
    best_params = None
    results = []

    # Iterate over all combinations of p, d, q
    for (p, d, q) in product(p_range, d_range, q_range):
        try:
            # Fit ARIMA model
            model = ARIMA(data, order=(p, d, q))
            model_fit = model.fit()

            # Store the chosen criterion
            if criterion == 'AIC':
                score = model_fit.aic
            elif criterion == 'BIC':
                score = model_fit.bic
            else:
                raise ValueError("Criterion must be either 'AIC' or 'BIC'")

            # Save the results
            results.append((p, d, q, score))

            # Update best parameters
            if score < best_score:
                best_score = score
                best_params = (p, d, q)
        except Exception as e:
            continue  # Skip models that fail to fit

    # Convert the results into a DataFrame for analysis
    results_df = pd.DataFrame(results, columns=['p', 'd', 'q', criterion])
    
    return best_params, best_score, results_df

# Example usage
np.random.seed(42)
n = 100
data = np.cumsum(np.random.randn(n))  # Simulated random walk

# Define the range of parameters
p_range = range(0, 3)  # AR terms
d_range = range(0, 2)  # Differencing terms
q_range = range(0, 3)  # MA terms

# Run grid search
best_params, best_score, results_df = optimize_arima(data, p_range, d_range, q_range, criterion='AIC')

print(f"Best Parameters: {best_params}, Best AIC: {best_score}")
print("Top 5 Results:")
print(results_df.sort_values('AIC').head())
