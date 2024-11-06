import numpy as np
from lib.kinematic_bycicle_model import KinematicBicycleModel

# Lớp Bộ lọc Kalman Mở rộng (EKF)
class ExtendedKalmanFilter:
    def __init__(self, wheelbase, length_rear, delta_time, P, Q, R):
        self.kbm = KinematicBicycleModel(wheelbase,length_rear, delta_time)
        self.length_rear = length_rear
        self.wheelbase = wheelbase
        self.delta_time = delta_time
        self.P = np.array(P, dtype=np.float32)
        self.Q = np.array(Q, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
    
    def jacobian(self, velocity, steering, heading):
        e = velocity
        l = steering
        w = self.length_rear
        h = self.wheelbase
        k = heading
        t = self.delta_time
        slip_angle = np.degrees(np.arctan(w * np.tan(np.radians(l)) / h))
        dF_dX = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [t * np.cos(np.radians(k + slip_angle)), -e * h * t * w * np.sin(np.radians(k + slip_angle)) * (1 / np.cos(np.radians(l)))**2 / (h**2 + w**2 * np.tan(np.radians(l))**2), 1, 0, -e * t * np.sin(np.radians(k + slip_angle))],
            [t * np.sin(np.radians(k + slip_angle)), e * h * t * w * np.cos(np.radians(k + slip_angle)) * (1 / np.cos(np.radians(l)))**2 / (h**2 + w**2 * np.tan(np.radians(l))**2), 0, 1, e * t * np.cos(np.radians(k + slip_angle))],
            [t * np.tan(np.radians(l)) * np.abs(h) / (h * np.sqrt(h**2 + w**2 * np.tan(np.radians(l))**2)), e * h**3 * t * (1 / np.cos(np.radians(l)))**2 / ((h**2 + w**2 * np.tan(np.radians(l))**2)**(3/2) * np.abs(h)), 0, 0, 1]
        ])
        return dF_dX

    def prediction_state(self, state, control, P, Q):
        velocity, steering, x, y, heading = state
        acceleration, steering_rate = control
        new_velocity, new_steering, new_x, new_y, new_heading = self.kbm.discrete_kbm(
            velocity, steering, x, y, heading, acceleration, steering_rate
        )
        J = self.jacobian(velocity, steering, heading)
        P_next = J @ P @ J.T + Q
        new_state = [new_velocity, new_steering, new_x, new_y, new_heading]
        return new_state, P_next

    def update_state(self, state, P, measurement, R):
        z = np.array(measurement, dtype=np.float32)
        H = np.array([
            [1, 0, 0, 0, 0],  # Measurement function for velocity
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],  # Measurement function for x
            [0, 0, 0, 1, 0],  # Measurement function for y
            [0, 0, 0, 0, 1]   # Measurement function for heading
        ], dtype=np.float32)
        
        # Calculate the measurement prediction
        z_pred = np.dot(H, state)
        
        # Calculate the measurement residual
        y_k = z - z_pred
        
        # Calculate the residual covariance
        S = np.dot(H, np.dot(P, H.T)) + R
        
        # Calculate the Kalman gain
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
        
        # Update the state estimate
        state_update = state + np.dot(K, y_k)
        
        # Update the covariance estimate
        I = np.eye(len(P), dtype=np.float32)
        P_update = np.dot(I - np.dot(K, H), P)
        
        return state_update, P_update

    def update_state_vel_head(self, state, P, measurement, R):
        z = np.array(measurement, dtype=np.float32)
        H = np.array([
            [1, 0, 0, 0, 0],  # Measurement function for velocity
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],  # Measurement function for x
            [0, 0, 0, 0, 0],  # Measurement function for y
            [0, 0, 0, 0, 1]   # Measurement function for heading
        ], dtype=np.float32)
        
        # Calculate the measurement prediction
        z_pred = np.dot(H, state)
        
        # Calculate the measurement residual
        y_k = z - z_pred
        
        # Calculate the residual covariance
        S = np.dot(H, np.dot(P, H.T)) + R
        
        # Calculate the Kalman gain
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))
        
        # Update the state estimate
        state_update = state + np.dot(K, y_k)
        
        # Update the covariance estimate
        I = np.eye(len(P), dtype=np.float32)
        P_update = np.dot(I - np.dot(K, H), P)
        
        return state_update, P_update