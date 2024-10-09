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
