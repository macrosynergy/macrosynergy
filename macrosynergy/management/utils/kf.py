import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        """
        Initialize the Kalman Filter
        
        :param F: State transition matrix
        :param H: Observation matrix
        :param Q: Process noise covariance
        :param R: Measurement noise covariance
        :param x0: Initial state estimate
        :param P0: Initial state covariance
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0
        self.log_likelihood = 0.0

    def predict(self):
        """
        Predict the next state and covariance.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        """
        Update the state estimate using the new measurement z.
        """
        y = z - self.H @ self.x  # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R  # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        # Update state and covariance
        self.x = self.x + K @ y
        self.P = self.P - K @ self.H @ self.P

        # Update log-likelihood
        self.log_likelihood += -0.5 * (np.log(np.linalg.det(S)) + y.T @ np.linalg.inv(S) @ y + np.log(2 * np.pi))

        return self.x, self.P
        


def kalman_smoother(F, x_filt, P_filt):
    """
    Kalman smoother implementation (RTS smoother).
    
    :param F: State transition matrix
    :param x_filt: Filtered state estimates from the Kalman filter
    :param P_filt: Filtered state covariances from the Kalman filter
    """
    n = len(x_filt)
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    
    # Set final step
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    for k in range(n-2, -1, -1):
        P_pred = F @ P_filt[k] @ F.T + Q
        G = P_filt[k] @ F.T @ np.linalg.inv(P_pred)
        
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k+1] - F @ x_filt[k])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k+1] - P_pred) @ G.T
        
    return x_smooth, P_smooth

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
import matplotlib.pyplot as plt

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