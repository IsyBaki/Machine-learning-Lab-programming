import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def run_task2():
    # Exponential growth model
    def exp_growth(t, a, b):
        return a * np.exp(b * t)

    time = np.array([0, 1, 2, 3, 4, 5])
    bacteria = np.array([100, 180, 324, 583.2, 1049.76, 1889.57])

    # Fit model
    params, _ = curve_fit(exp_growth, time, bacteria, p0=(100, 0.5))
    a_fit, b_fit = params

    print("\n--- Task 2: Exponential Fit ---")
    print(f"a (initial count): {a_fit:.4f}")
    print(f"b (growth rate): {b_fit:.4f}")

    # Predicted values
    bacteria_pred = exp_growth(time, a_fit, b_fit)

    # R² calculation
    ss_res = np.sum((bacteria - bacteria_pred) ** 2)
    ss_tot = np.sum((bacteria - np.mean(bacteria)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R²: {r_squared:.4f}")

    # Plot
    t_cont = np.linspace(0, 5, 100)
    y_fit = exp_growth(t_cont, a_fit, b_fit)

    plt.figure(figsize=(8, 6))
    plt.scatter(time, bacteria, color='red', label='Data Points')
    plt.plot(t_cont, y_fit, label=f'Fitted Curve: y={a_fit:.2f}e^({b_fit:.2f}t)')
    plt.title('Exponential Growth of Bacteria')
    plt.xlabel('Time (hours)')
    plt.ylabel('Bacteria Count')
    plt.legend()
    plt.grid(True)
    plt.show()
