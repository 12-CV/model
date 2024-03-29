import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = "output.txt"    # 데이터 파일 경로

data = np.loadtxt(file_path)

x = data[:, 0]
y = data[:, 1]
real_distances = data[:, 2]

def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c

params, _ = curve_fit(quadratic_func, x, y)

# 근사식 출력
a, b, c = params
equation = f"{a:.2f}x^2 + {b:.2f}x + {c:.2f}"
print("Quadratic approximation equation:", equation)

x_fit = np.linspace(min(x), max(x), 100)
y_fit = quadratic_func(x_fit, *params)

plt.scatter(x, y, label='Data')
plt.plot(x_fit, y_fit, color='red', label='Quadratic Approximation')
plt.title('Quadratic Approximation')
plt.xlabel('Depth')
plt.ylabel('Distance')
plt.legend()
plt.savefig('quadratic_approximation.png', bbox_inches='tight')

# 정확도 계산
predicted_y_values = quadratic_func(x, *params)
accuracies = np.abs(predicted_y_values - real_distances)
average_accuracy = np.mean(accuracies)
print("Average accuracy:", average_accuracy)
