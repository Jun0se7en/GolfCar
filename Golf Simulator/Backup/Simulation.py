import pandas as pd
import numpy as np
import cv2
from lib.extended_kalman_filter import ExtendedKalmanFilter
from lib.Transform import Transform
import matplotlib.pyplot as plt

transform = Transform()
time = 0.0
gps_latitude = 0.0
gps_longitude = 0.0
prev_gps_latitude = 0.0
prev_gps_longitude = 0.0
heading = 0.0
acc = 0.0
steer = 0
speed = 0
steering_rate = 0.0
wheelbase = 2.1
length_rear = 1.0
dt = 0.01
initial_flag = False
P_ekf = np.array([
    [0.1, 0, 0, 0, 0], # Speed
    [0, 1e-10, 0, 0, 0], # Steer
    [0, 0, 2, 0, 0], # X
    [0, 0, 0, 2, 0], # Y
    [0, 0, 0, 0, 0.1], # Heading
])
Q_ekf = np.array([
    [0.01, 0, 0, 0, 0],
    [0, 0.01, 0, 0, 0],
    [0, 0, 0.01, 0, 0],
    [0, 0, 0, 0.01, 0],
    [0, 0, 0, 0, 0.01],
])
R_ekf = np.array([
    [0.1, 0, 0, 0, 0],
    [0, 1e-10, 0, 0, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 2, 0],
    [0, 0, 0, 0, 0.01],
])
ekf = ExtendedKalmanFilter(wheelbase, length_rear, dt, P_ekf, Q_ekf, R_ekf)

filter_x = []
filter_y = []
gps_x = []
gps_y = []
current_state = [0, 0, 0, 0, 0]
current_covariance = P_ekf
predicted_state = current_state
predicted_P = current_covariance

if __name__ == "__main__":
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
    # Labels and titles for each subplot
    ax1.set_title('GPS Coordinates')
    ax1.set_xlabel('gps_x')
    ax1.set_ylabel('gps_y')
    ax1.grid(True)

    ax2.set_title('Filtered Coordinates')
    ax2.set_xlabel('filter_x')
    ax2.set_ylabel('filter_y')
    ax2.grid(True)
    # Mở camera, thông thường 0 là ID của webcam mặc định
    cap = cv2.VideoCapture('./RetrievingVideo6.mp4')
    index = 0
    df = pd.read_csv('./Retrieving_Data6.csv')
    data = df.values

    # Kiểm tra nếu camera được mở thành công
    if not cap.isOpened():
        print("Không thể mở camera")
        exit()

    # Vòng lặp để đọc và hiển thị từng khung hình từ camera
    while True:
        # Đọc một khung hình từ camera
        ret, frame = cap.read()

        # Nếu không đọc được, thoát vòng lặp
        if not ret:
            print("Không thể nhận khung hình (có thể camera đã bị ngắt kết nối).")
            break

        gps_latitude = data[index][1]
        gps_longitude = data[index][2]
        lat = gps_latitude
        lon = gps_longitude
        alt = 0.0
        target_lla = [lat, lon, alt]
        ned = transform.lla_to_ned(target_lla)
        speed = data[index][3]
        delta_time = data[index][0] - time
        steering_rate = (data[index][4] - steer) / (delta_time)
        steer = data[index][4]
        heading = data[index][5]
        if heading > 0:
            heading = -180-(180-heading)
        acc = data[index][6]
        time = data[index][0]
        
        print("GPS_X:", ned[1], "GPS_Y:", ned[0], "Speed:", speed, "Steer:", steer, "Heading:", heading, "Acc:", acc, "StR:", steering_rate, f"dt: {delta_time:.3f}")
        if not initial_flag:
            if int(gps_latitude) != 0 and int(gps_longitude) != 0:
                lat = gps_latitude
                lon = gps_longitude
                alt = 0.0
                target_lla = [lat, lon, alt]
                ned = transform.lla_to_ned(target_lla)
                gps_x.append(ned[1])
                gps_y.append(ned[0])
                current_state = [speed, steer, ned[0], ned[1], heading]  # [velocity, steering, x, y, heading]
                current_covariance = P_ekf
                prev_gps_latitude = gps_latitude
                prev_gps_longitude = gps_longitude
                initial_flag = True
        else:
            if speed > 0:
                current_state, current_covariance = ekf.prediction_state(current_state, [acc, steering_rate], current_covariance, Q_ekf)
                # current_state[1] = steer
                if prev_gps_latitude != gps_latitude and prev_gps_longitude != gps_longitude and int(gps_latitude) != 0 and int(gps_longitude) != 0:
                    lat = gps_latitude
                    lon = gps_longitude
                    alt = 0.0
                    target_lla = [lat, lon, alt]
                    ned = transform.lla_to_ned(target_lla)
                    gps_x.append(ned[1])
                    gps_y.append(ned[0])
                    measurement = np.array([speed, steer, ned[0], ned[1], heading])
                    current_state, current_covariance = ekf.update_state(current_state, current_covariance, measurement, R_ekf)
                else:
                    measurement = np.array([speed, steer, 0, 0, heading])
                    current_state, current_covariance = ekf.update_state_vel_head(current_state, current_covariance, measurement, R_ekf)
                # current_state[1] = steer    
                print('Predicted State:', current_state)
                ned = [current_state[2], current_state[3], 0]
                filter_x.append(ned[1])
                filter_y.append(ned[0])


        # Hiển thị khung hình
        cv2.imshow('Camera', cv2.resize(frame, (640, 320)))
        index += 1
        # Re-plot the GPS coordinates
        ax1.set_xlim(-300, 300)  # Set x-axis limits for GPS plot
        ax1.set_ylim(-300, 300)  # Set y-axis limits for GPS plot
        ax1.plot(gps_x, gps_y, 'bo-', label="GPS", markersize=0.1)
        ax1.set_title('GPS Coordinates')
        ax1.set_xlabel('gps_x')
        ax1.set_ylabel('gps_y')
        ax1.grid(True)

        # Re-plot the filtered coordinates
        ax2.set_xlim(-300, 300)  # Set x-axis limits for Filtered plot
        ax2.set_ylim(-300, 300)  # Set y-axis limits for Filtered plot
        ax2.plot(filter_x, filter_y, 'ro-', label="Filtered", markersize=0.1)
        ax2.set_title('Filtered Coordinates')
        ax2.set_xlabel('filter_x')
        ax2.set_ylabel('filter_y')
        ax2.grid(True)

        # Pause to update the plots
        plt.pause(0.01)
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.ioff()
    plt.show()
    # Giải phóng camera sau khi hoàn tất
    cap.release()

    # Đóng tất cả các cửa sổ hiển thị
    cv2.destroyAllWindows()
