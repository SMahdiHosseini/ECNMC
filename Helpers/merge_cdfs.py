import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import Counter

def compute_cdf_Y_from_file(csv_file, num_samples=100000):
    # Read CDF of X from CSV file
    df = pd.read_csv(csv_file)
    x_values = df['packet_size'].astype(int).values  # Ensure X is integer
    cdf_X = df['cdf'].values
    
    # Generate random samples from X using inverse transform sampling
    uniform_samples = np.random.rand(num_samples)
    X_samples = np.interp(uniform_samples, cdf_X, x_values).astype(int)
    
    # Generate two independent samples
    X1_samples = np.random.choice(X_samples, num_samples)
    X2_samples = np.random.choice(X_samples, num_samples)
    
    # Compute Y = X1 - X2 (integer values)
    Y_samples = X1_samples - X2_samples
    
    # Count occurrences to estimate CDF
    y_counts = Counter(Y_samples)
    y_sorted = np.array(sorted(y_counts.keys()))
    cdf_Y = np.cumsum([y_counts[y] for y in y_sorted]) / num_samples
    
    return y_sorted, cdf_Y

# Example usage
csv_file = "packet_size_cdf_singleQueue.csv"  # Replace with actual file path
y_values, cdf_Y = compute_cdf_Y_from_file(csv_file)

# save the cdf_Y to a file
cdf_Y_df = pd.DataFrame({'packet_size': y_values, 'cdf': cdf_Y})
cdf_Y_df.to_csv('cdf_y.csv', index=False)
