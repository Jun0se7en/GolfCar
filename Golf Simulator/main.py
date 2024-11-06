#####
## SPINN DESIGN CODE
# YOUTUBE: (SPINN TV) https://www.youtube.com/spinnTv
# WEBSITE: spinncode.com
########################################################################

########################################################################
## IMPORTS
########################################################################
import os
import sys
import io

from PySide6.QtWidgets import QMainWindow, QApplication, QGraphicsColorizeEffect
from PySide6.QtGui import QFontDatabase
from PySide6.QtCore import QThread, QEvent, Signal

import folium

import cv2
import socket
import pickle
import struct

from copy import copy

import csv
from time import sleep
from threading import Thread
import signal

import base64
import numpy as np
import json

################################################################################################
# Import the generated UI
################################################################################################
# from ui_interface import *
from ui_MainWindow import *

################################################################################################
# MAIN WINDOW CLASS
################################################################################################
class MainWindow(QMainWindow):     
    def __init__(self, parent=None):
        QMainWindow.__init__(self)

        ################################################################################################
        # Setup the flags for threads
        ################################################################################################
        self.RunCamera = True
        self.RunMapView = True
        self.RunOutputImage = True

        ################################################################################################
        # Setup the UI main window
        ################################################################################################
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        ################################################################################################
        # Show window
        ################################################################################################
        self.show()

        # ################################################################################################
        # # Prepare web data
        # ################################################################################################
        # # Create a folium map object centered at the location
        # self.map = folium.Map(location=[10.87043, 106.80196], zoom_start=17)

        # m = copy(self.map)

        # # Add a marker for a specific GPS point (initial position)
        # default_coords = (10.86963, 106.80384)
        # folium.Marker(default_coords, popup='Voi Coffee').add_to(m)

        # # save map data to data object
        # data = io.BytesIO()
        # m.save(data, close_file=False)

        # self.ui.WebView.setHtml(data.getvalue().decode())

        ################################################################################################
        # HIDE LEFT/RIGHT ARROW SIGNAL
        ################################################################################################
        self.ui.lb_Left_Signal.setVisible(False)
        self.ui.lb_Right_Signal.setVisible(False)

        ################################################################################################
        # CUSTOMIZE ANALOGUE GAUGE SPEED WIDGET
        ################################################################################################
        self.ui.Analog_Gauge_Speed.enableBarGraph = True
        self.ui.Analog_Gauge_Speed.valueNeedleSnapzone = 1

        ################################################################################################
        # Set Angle Offset
        ################################################################################################
        Speed_Gauge_Offset = 0
        Battery_Gauge_Offset = 0
        self.ui.Analog_Gauge_Speed.updateAngleOffset(Speed_Gauge_Offset)
        # self.ui.Analog_Gauge_Battery.updateAngleOffset(-Battery_Gauge_Offset)

        ################################################################################################
        # Set gauge units
        ################################################################################################
        self.ui.Analog_Gauge_Speed.units = "Km/h"
        # self.ui.Analog_Gauge_Battery.units = "Power"

        ################################################################################################
        # Set minimum gauge value
        ################################################################################################
        self.ui.Analog_Gauge_Speed.minValue = 0
        # self.ui.Analog_Gauge_Battery.minValue = 0
        ################################################################################################
        # Set maximum gauge value
        ################################################################################################
        self.ui.Analog_Gauge_Speed.maxValue = 100
        # self.ui.Analog_Gauge_Battery.maxValue = 100

        ################################################################################################
        # Set scale divisions
        ################################################################################################
        self.ui.Analog_Gauge_Speed.scalaCount = 10
        # self.ui.Analog_Gauge_Battery.scalaCount = 10

        # Start from the minimum value
        # self.ui.widget.updateValue(self.ui.widget.minValue)
        # OR
        # Start from half/middle value
        self.ui.Analog_Gauge_Speed.updateValue(int(self.ui.Analog_Gauge_Speed.maxValue - self.ui.Analog_Gauge_Speed.minValue)/2)
        # self.ui.Analog_Gauge_Battery.updateValue(int(self.ui.Analog_Gauge_Battery.maxValue - self.ui.Analog_Gauge_Battery.minValue)/2)

        ################################################################################################
        # Select gauge theme
        ################################################################################################
        #self.ui.Analog_Gauge_Speed.setGaugeTheme(20)

        # self.ui.widget.setCustomGaugeTheme(
        #     color1 = "#FF2B00",
        #     color2= "#821600",
        #     color3 = "#260600"
        # )

        self.ui.Analog_Gauge_Speed.setCustomGaugeTheme(
            color1 = "red",
            color2 = "orange",
            color3 = "green"
        )

        # self.ui.Analog_Gauge_Battery.setCustomGaugeTheme(
        #     color1 = "green",
        #     color2 = "orange",
        #     color3 = "red"
        # )

        # self.ui.widget.setCustomGaugeTheme(
        #     color1 = "#fff",
        #     color2= "#555",
        #     color3 = "#000"
        # )

        # self.ui.widget.setScalePolygonColor(
        #     color1 = "gray"
        # )

        self.ui.Analog_Gauge_Speed.setNeedleCenterColor(
            color1 = "dark gray"
        )

        # self.ui.Analog_Gauge_Battery.setNeedleCenterColor(
        #     color1 = "light black"
        # )

        self.ui.Analog_Gauge_Speed.setOuterCircleColor(
            color1 = "dark gray"
        )

        # self.ui.Analog_Gauge_Battery.setOuterCircleColor(
        #     color1 = "dark gray"
        # )

        self.ui.Analog_Gauge_Speed.setBigScaleColor("yellow")
        self.ui.Analog_Gauge_Speed.setFineScaleColor("blue")

        host = "192.168.2.193"
        port = 12345

        # if self.RunCamera:
        #     self.CameraWorker = UpdateCameraWorker(host, port)
        #     self.CameraWorker.start()
        #     self.CameraWorker.ImageUpdate.connect(self.ImageUpdateSlot)

        # if self.RunOutputImage:
        #     self.OutputImageWorker = UpdateCameraWorker(host, port+1)
        #     self.OutputImageWorker.start()
        #     self.OutputImageWorker.ImageUpdate.connect(self.OutputImageUpdateSlot)

        if self.RunMapView:
            self.initial_location = [10.87043, 106.80196]
            self.MapWorker = UpdateMapWorker(self.initial_location)
            self.MapWorker.start()
            self.MapWorker.MapUpdate.connect(self.WebviewUpdateSlot)

        #Register signal for program exit!
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        # logger.info("Received Ctrl+C, stopping processes...")
        # Stop all processes:
        if self.RunCamera:
            self.CameraWorker.stop()
        if self.RunMapView:
            self.MapWorker.stop()
        sys.exit(0)

    def ImageUpdateSlot(self, image):
        self.ui.lb_Raw_Img.setPixmap(QPixmap.fromImage(image))

    def OutputImageUpdateSlot(self, image):
        self.ui.lb_Output_Img.setPixmap(QPixmap.fromImage(image))

    def WebviewUpdateSlot(self, map_data):
        # Update the WebView with the updated map data
        self.ui.WebView.setHtml(map_data.decode())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.ui.lb_Left_Signal.setVisible(True)   # Show the left arrow signal
        elif event.key() == Qt.Key.Key_Right:
            self.ui.lb_Right_Signal.setVisible(True)  # Show the right arrow signal
            
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.ui.lb_Left_Signal.setVisible(False)   # Hide the left arrow signal
        elif event.key() == Qt.Key.Key_Right:
            self.ui.lb_Right_Signal.setVisible(False)  # Hide the right arrow signal

'''
        ################################################################################################
        # Select gauge theme
        ################################################################################################
        # self.ui.widget.setGaugeTheme(0)

        # self.ui.widget.setCustomGaugeTheme(
        #     color1 = "#FF2B00",
        #     color2= "#821600",
        #     color3 = "#260600"
        # )

        self.ui.widget.setCustomGaugeTheme(
            color1 = "#002523",
            color2= "#990008",
            color3 = "#00F6E9"
        )

        # self.ui.widget.setCustomGaugeTheme(
        #     color1 = "#fff",
        #     color2= "#555",
        #     color3 = "#000"
        # )

        # self.ui.widget.setScalePolygonColor(
        #     color1 = "red"
        # )

        # self.ui.widget.setNeedleCenterColor(
        #     color1 = "red"
        # )

        # self.ui.widget.setOuterCircleColor(
        #     color1 = "red"
        # )

        self.ui.widget.setBigScaleColor("#005275")
        self.ui.widget.setFineScaleColor("#005275")

        for x in range(1, 25):
            self.ui.theme_comboBox.addItem(str(x))


        ################################################################################################
        # Set custom font
        ################################################################################################
        QFontDatabase.addApplicationFont(os.path.join(os.path.dirname(__file__), 'fonts/ds_digital/DS-DIGIB.TTF') )

        # self.ui.widget.setValueFontFamily("DS-Digital")
        # self.ui.widget.setScaleFontFamily("Verdana")

        ################################################################################################
        # Change gauge theme
        ################################################################################################
        self.ui.theme_comboBox.currentTextChanged.connect(lambda: self.changeGaugeTheme())

        self.updateGaugeValue()       

        ################################################################################################
        # END
        ################################################################################################

    def changeGaugeTheme(self):
        self.ui.widget.setGaugeTheme(int(self.ui.theme_comboBox.currentText()))

    def updateScalaCount(self):
        self.ui.widget.setScalaCount(10)

    def updateMaxVal(self):
        self.ui.widget.setMaxValue(100)
        
    def updateMinVal(self):
        self.ui.widget.setMinValue(0)

    def updateAngleOffset(self):
        self.ui.widget.updateAngleOffset(0)

    def updateGaugeColorOuterRadius(self):
        self.ui.widget.setGaugeColorOuterRadiusFactor(10)    

    def updateGaugeColorInnerRadius(self):
        self.ui.widget.setGaugeColorInnerRadiusFactor(10)      

    def updateStartScaleAngle(self):
        self.ui.widget.setScaleStartAngle(10)

    def updateScaleAngleSize(self):
        self.ui.widget.setTotalScaleAngleSize(10)

    def updateGaugeValue(self):
        self.ui.widget.updateValue(10)

    ################################################################################################
    # SET NEEDLE COLOR
    ################################################################################################
    def setNeedleColor(self):
        # Get RGBA values
        R = 100
        G = 0
        B = 0
        Transparency = 50
        # print(R, G, B, Transparency)
        self.ui.widget.setNeedleColor(R=R, G=G, B=B, Transparency=Transparency)

    ################################################################################################
    # SET NEEDLE COLOR ON DRAG
    ################################################################################################
    def setNeedleColorOnDrag(self):
        # Get RGBA values
        R = 0
        G = 100
        B = 0
        Transparency = 50
        # print(R, G, B, Transparency)
        self.ui.widget.setNeedleColorOnDrag(R=R, G=G, B=B, Transparency=Transparency)


    ################################################################################################
    # SET SCALE TEXT COLOR
    ################################################################################################
    def setScaleValueColor(self):
        # Get RGBA values
        R = 0
        G = 0
        B = 100
        Transparency = 50
        # print(R, G, B, Transparency)
        self.ui.widget.setScaleValueColor(R=R, G=G, B=B, Transparency=Transparency)


    ################################################################################################
    # SET VALUE DISPLAY COLOR
    ################################################################################################
    def setDisplayValueColor(self):
        # GET RGBA VALUE
        R = 50
        G = 50
        B = 50
        Transparency = 50
        # print(R, G, B, Transparency)
        self.ui.widget.setDisplayValueColor(R=R, G=G, B=B, Transparency=Transparency)

    ################################################################################################
    # SHOW HIDE VALUE TEXT
    ################################################################################################
    def en_disable_ValueText(self):
        self.ui.widget.setEnableValueText(True)

    ################################################################################################
    # SHOW HIDE CENTER POINTER
    ################################################################################################
    def en_disable_CB_CenterPoint(self):
        self.ui.widget.setEnableCenterPoint(True)

    ################################################################################################
    # SHOW HIDE NEEDLE
    ################################################################################################
    def en_disable_Needle(self):
        self.ui.widget.setEnableNeedlePolygon(True)

    ################################################################################################
    # SHOW HIDE SCALE TEXT
    ################################################################################################
    def en_disable_ScaleText(self):
        self.ui.widget.setEnableScaleText(True)

    ################################################################################################
    # ENABLE DISABLE FILL COLOR
    ################################################################################################
    def setEnableScalePolygon(self):
        self.ui.widget.setEnableScalePolygon(True)

    ################################################################################################
    # ENABLE DISABLE BIG SCALE
    ################################################################################################
    def set_enable_Scale_Grid(self):
        self.ui.widget.setEnableBigScaleGrid(True)

    ################################################################################################
    # ENABLE DISABLE FINE SCALE MARKERS
    ################################################################################################
    def set_enable_fine_Scale_Grid(self):
        self.ui.widget.setEnableFineScaleGrid(True)
'''

class UpdateCameraWorker(QThread):
    ImageUpdate = Signal(QImage)

    def __init__(self, host, port):
        super().__init__()
        self.ThreadActive = True
        self.host = host
        self.port = port
        

    def run(self):
        self.ThreadActive = True

        # host = socket.gethostname()  # as both code is running on same pc
        # host = "192.168.2.193"
        # port = 12345  # socket server port number

        client_socket = socket.socket()  # instantiate
        print('Socket created')

        client_socket.connect((self.host, self.port))  # connect to the server

        data = b""
        payload_size = struct.calcsize("Q")       

        while self.ThreadActive:
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024)
                if not packet: break
                data+=packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q",packed_msg_size)[0]
            
            while len(data) < msg_size:
                data += client_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data  = data[msg_size:]
            frame = pickle.loads(frame_data)
            # cv2.imshow("RECEIVING VIDEO",frame)
            image_data = base64.b64decode(frame)
            img = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(img, cv2.IMREAD_COLOR)
            key = cv2.waitKey(1) & 0xFF
            if key  == 27:
                break

            # Get the frame height, width and channels.
            height, width, channels = frame.shape
            # Calculate the number of bytes per line.
            bytes_per_line = width * channels
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FlippedImage = cv2.flip(Image, 1)
            # Convert the image to Qt format.
            qt_rgb_image = QImage(FlippedImage.data, width, height, bytes_per_line, QImage.Format_RGB888)
            # Scale the image.
            # NOTE: consider removing the flag Qt.KeepAspectRatio as it will crash Python on older Windows machines
            # If this is the case, call instead: qt_rgb_image.scaled(1280, 720) 
            qt_rgb_image_scaled = qt_rgb_image.scaled(320, 240, Qt.KeepAspectRatio)  # 720p
            # qt_rgb_image_scaled = qt_rgb_image.scaled(1920, 1080, Qt.KeepAspectRatio)
            # Emit this signal to notify that a new image or frame is available.
            self.ImageUpdate.emit(qt_rgb_image_scaled)
        # When everything done, release the socket
        client_socket.close()
        # Tells the thread's event loop to exit with return code 0 (success).
        self.terminate()

    def stop(self):
        self.ThreadActive = False
        self.terminate()

class UpdateMapWorker(QThread):
    MapUpdate = Signal(bytes)

    def __init__(self, initial_location):
        super().__init__()
        self.ThreadActive = True
        self.initial_location = initial_location

        self.map = folium.Map(location=self.initial_location, zoom_start=17)
        self.marker = folium.Marker(location=self.initial_location)
        self.marker.add_to(self.map)
        self.filter_coordinates = []
        self.gps_coordinates = []
        self.SERVER_ADDRESS = '192.168.2.171'
        self.FILTER_PORT = 3333
        self.GPS_PORT = 4444


    def run(self):
        self.ThreadActive = True
        # self.map = folium.Map(location=[10.87043, 106.80196], zoom_start=17)
        self.filter_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Filter Socket created')
        self.filter_sock.connect((self.SERVER_ADDRESS, self.FILTER_PORT))
        self.gps_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('GPS Socket created')
        self.gps_sock.connect((self.SERVER_ADDRESS, self.GPS_PORT))
        while self.ThreadActive:
            ### GPS ###
            self.gps_request = 'GET_GPS'
            # print(f'Sending request: {request}')
            self.gps_sock.sendall(self.gps_request.encode())

            # Receive response from the server
            gps_data = self.gps_sock.recv(1024)
            gps_response = gps_data.decode()
            # print(f'Received response: {response}')

            # Parse the JSON data
            try:
                json_data = json.loads(gps_response)
                self.gps_latitude = json_data['GPS_Lat']
                self.gps_longitude = json_data['GPS_Long']

                print(f'GPS_Latitude: {self.gps_latitude}')
                print(f'GPS_Longitude: {self.gps_longitude}')

            except json.JSONDecodeError:
                print('Error: Invalid JSON data received from the server')
            self.gps_coordinates.append((self.gps_latitude, self.gps_longitude))
            try:
                with open('./GPS.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.gps_coordinates[-1])
            except:
                with open('./GPS.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.gps_coordinates[-1])
            # Create a PolyLine object with the coordinates
            folium.PolyLine(locations=self.gps_coordinates, color='blue').add_to(self.map)



            ### Filter ###
            self.filter_request = 'GET_JSON_DATA'
            # print(f'Sending request: {request}')
            self.filter_sock.sendall(self.filter_request.encode())

            # Receive response from the server
            data = self.filter_sock.recv(1024)
            response = data.decode()
            # print(f'Received response: {response}')

            # Parse the JSON data
            try:
                json_data = json.loads(response)
                self.speed = json_data['speed']
                self.latitude = json_data['lat']
                self.longitude = json_data['long']
                self.heading = json_data['heading']

                print(f'Speed: {self.speed*3.6} km/h')
                print(f'Latitude: {self.latitude}')
                print(f'Longitude: {self.longitude}')
                print(f'Heading: {self.heading} degrees')

            except json.JSONDecodeError:
                print('Error: Invalid JSON data received from the server')
            self.filter_coordinates.append((self.latitude, self.longitude))
            try:
                with open('./Filter.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.filter_coordinates[-1])
            except:
                with open('./Filter.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(self.filter_coordinates[-1])
            # Create a PolyLine object with the coordinates
            folium.PolyLine(locations=self.filter_coordinates, color='red').add_to(self.map)

                # Save map data to data object
            data = io.BytesIO()
            self.map.save(data, close_file=False)

            # Emit the map_updated signal with the updated map data
            self.MapUpdate.emit(data.getvalue())

            # Wait for 1 second before updating the next coordinate
            sleep(1)
        self.sock.close()
        self.terminate()

    def stop(self):
        self.ThreadActive = False
        self.terminate()


########################################################################
## EXECUTE APP
########################################################################
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ########################################################################
    ## 
    ########################################################################
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

########################################################################
## END===>
########################################################################  