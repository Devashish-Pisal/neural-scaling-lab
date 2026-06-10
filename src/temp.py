import matplotlib.pyplot as plt
import numpy as np



x = np.array([0.10, 0.25, 0.50, 1.00])
y = np.array([18.2, 11.0, 5.2, 2.4])


logx = np.log10(x)
logy = np.log10(y)
coeffs = np.polyfit(logx, logy, 1)
k, loga = coeffs
a = 10**loga


x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
y_fit = a * x_fit**k


plt.figure(figsize=(8, 6))
plt.loglog(x, y, 'bo-', label='Data')
plt.loglog(x_fit, y_fit, 'r--', label=f'Fit: $y = {a:.2f} x^{{{k:.2f}}}$')
plt.xlabel('X (log scale)')
plt.ylabel('Y (log scale)')
plt.title('Scaling Law Plot with Power-Law Fit')
plt.grid(True, which="both", ls="--")
plt.legend()

plt.show()


print(f"Fitted scaling law: y = {a:.2f} * x^{k:.2f}")