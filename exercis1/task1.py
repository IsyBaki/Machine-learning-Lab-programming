import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_task1():
    data = pd.read_csv('HeightWeight.csv')
    heights = data['Height(Inches)'].to_numpy()
    weights = data['Weight(Pounds)'].to_numpy()

    mean_height = np.mean(heights)
    mean_weight = np.mean(weights)
    std_height = np.std(heights, ddof=1)
    std_weight = np.std(weights, ddof=1)

    print("\n--- Task 1: Statistical Values ---")
    print(f"Average height (inches): {mean_height:.2f}")
    print(f"Average weight (pounds): {mean_weight:.2f}")
    print(f"Std deviation height: {std_height:.2f}")
    print(f"Std deviation weight: {std_weight:.2f}")

    # Plot histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=20, edgecolor='black')
    plt.title('Distribution of Heights')
    plt.xlabel('Height (inches)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(weights, bins=20, edgecolor='black', color='orange')
    plt.title('Distribution of Weights')
    plt.xlabel('Weight (pounds)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Correlation
    n = len(heights)
    covariance = np.sum((heights - mean_height) * (weights - mean_weight)) / (n - 1)
    correlation = covariance / (std_height * std_weight)

    print("\nCorrelation between Height and Weight:")
    print(f"Covariance: {covariance:.4f}")
    print(f"Correlation: {correlation:.4f}")
