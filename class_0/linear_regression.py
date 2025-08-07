import numpy as np
import matplotlib.pyplot as plt


# This code generates points along a line or a curve and adds gaussian noise. It then estimates the coefficients of the
# best fitting line to the noisy points
# using linear regression equations and plots the noisy points, true line/curve and estimated line

def main(f):

    def gen_y_linear(x, m, b):
        return m * x + b

    def gen_y_quadratic(x, a, b, c):
        return a * x * x + b * x + c

    n = 100
    x = np.linspace(0, 10, n)  # 100 points between 0 and 10

    m = 2  # Slope
    b = 5  # Y-intercept
    # linear
    if f == 'quadratic':
        # quadratic
        y_ideal = gen_y_quadratic(x, 1, 2, 3)
    else:
        y_ideal = gen_y_linear(x, m, b)


    mean = 0
    std_dev = 1.5  # Adjust for more or less noise
    noise = np.random.normal(mean, std_dev, size=len(x))

    # y_noisy is what we are given. We don't know y_ideal
    y_noisy = y_ideal + noise

    # Estimate the values of m and b
    sigma_x = np.sum(x)
    sigma_y = np.sum(y_noisy)
    sigma_x_sq = np.dot(x, x)

    sigma_xy = np.sum(x * y_noisy) # element wise multiplication
    m_est = (sigma_xy - sigma_y * sigma_x/n)/(sigma_x_sq - sigma_x * sigma_x/n)
    b_est = (sigma_y - m_est * sigma_x)/n
    # calculate using matrix notation
    x_ = np.concatenate((np.expand_dims(x, axis=0), np.ones((1,n))), axis=0)
    A_est = np.expand_dims(y_noisy, axis=0) @ x_.T @ np.linalg.inv(x_ @ x_.T)

    y_est = m_est * x + b_est

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_ideal, color='green', label='f(x)')
    plt.plot(x, y_est, color='blue', label='Estimated Line')
    plt.scatter(x, y_noisy, color='red', s=10, label='Noisy Points')
    plt.title('Curve with Gaussian Noise' if f == 'quadratic' else 'Line with Gaussian Noise')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    f = "linear"
    main(f)