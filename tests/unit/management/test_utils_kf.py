import unittest

import matplotlib.pyplot as plt
import numpy as np

from macrosynergy.management.utils.kf import KalmanFilter, kalman_smoother

class TestAll(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_kf(self):
        pass


# Test the Kalman filter and smoother with a simple example
# Define the model parameters
dt = 1.0
F = np.array([[1, dt], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])           # Observation matrix
Q = np.array([[0.001, 0], [0, 0.001]])  # Process noise covariance
R = np.array([[0.01]])                  # Measurement noise covariance
x0 = np.array([0, 1])            # Initial state estimate
P0 = np.eye(2)                   # Initial covariance estimate

# Simulate some measurements
np.random.seed(42)
n_steps = 50
true_x = np.zeros((n_steps, 2))
measurements = np.zeros(n_steps)

for i in range(1, n_steps):
    true_x[i] = F @ true_x[i-1] + np.random.multivariate_normal([0, 0], Q)
    measurements[i] = H @ true_x[i] + np.random.normal(0, np.sqrt(R))

# Run the Kalman filter
kf = KalmanFilter(F, H, Q, R, x0, P0)
x_filt = np.zeros((n_steps, 2))
P_filt = np.zeros((n_steps, 2, 2))

for t in range(n_steps):
    x_pred, P_pred = kf.predict()
    x_filt[t], P_filt[t] = kf.update(measurements[t])

# TODO minimise or maximise the log-likelihood?
print("Log-Likelihood:", kf.log_likelihood)

# Apply the Kalman smoother
x_smooth, P_smooth = kalman_smoother(F, x_filt, P_filt)

# Display the filtered and smoothed estimates
plt.figure(figsize=(10, 6))
plt.plot(true_x[:, 0], label='True Position', linestyle='--')
plt.plot(measurements, label='Measurements', linestyle=':')
plt.plot(x_filt[:, 0], label='Filtered Estimate')
plt.plot(x_smooth[:, 0], label='Smoothed Estimate', linestyle='-')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter and Smoother')
plt.show()