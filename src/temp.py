import matplotlib.pyplot as plt
import numpy as np


def plot_temp_scaling_law():
    x = np.array([0.10, 0.25, 0.50, 1.00])
    y = np.array([18.2, 11.0, 5.2, 2.4])
    # Fit in natural-log space
    logx = np.log(x)
    logy = np.log(y)
    k, loga = np.polyfit(logx, logy, 1)
    a = np.exp(loga)
    # Evaluate fit at the original x values
    y_fit = a * x**k
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(x, y, 'bo-', label='Data')
    ax.loglog(
        x,
        y_fit,
        'r--o',
        label=rf'Fit: $y = {a:.2f}x^{{{k:.2f}}}$'
    )
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Metric')
    ax.set_title('Scaling Law (Natural Log Fit)')
    ax.grid(True, which='both', ls='--')
    ax.legend()
    plt.show()
    print(f"Fitted scaling law: y = {a:.4f} * x^{k:.4f}")



def get_crossover_epochs():
    pass

if __name__ == "__main__":
    plot_temp_scaling_law()