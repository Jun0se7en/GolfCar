# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interfaceVewFiN.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
    QRadialGradient)
from PyQt5.QtWidgets import *

from AnalogGaugeWidget import AnalogGaugeWidget

from PyQt5.QtWebEngineWidgets import QWebEngineView


class Ui_MainWindow(object):
    def __init__(self, width = 1280, height = 720):
        self.width = width
        self.height = height
    def setupUi(self, MainWindow):
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(self.width, self.height)
        icon = QIcon()
        icon.addFile(u"images/Logo_UIT.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QSize(1280, 720))
        self.Header_Frame = QFrame(self.centralwidget)
        self.Header_Frame.setObjectName(u"Header_Frame")
        self.Header_Frame.setGeometry(QRect(0, 0, self.width, 80))
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.Header_Frame.sizePolicy().hasHeightForWidth())
        self.Header_Frame.setSizePolicy(sizePolicy1)
        self.Header_Frame.setFrameShape(QFrame.StyledPanel)
        self.Header_Frame.setFrameShadow(QFrame.Raised)
        self.Header_Frame.setLineWidth(0)
        self.Header_Frame_Layout = QHBoxLayout(self.Header_Frame)
        self.Header_Frame_Layout.setSpacing(0)
        self.Header_Frame_Layout.setObjectName(u"Header_Frame_Layout")
        self.Header_Frame_Layout.setContentsMargins(0, 0, 0, 0)
        self.Header_Left = QFrame(self.Header_Frame)
        self.Header_Left.setObjectName(u"Header_Left")
        self.Header_Left.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.Header_Left.sizePolicy().hasHeightForWidth())
        self.Header_Left.setSizePolicy(sizePolicy1)
        self.Header_Left.setFrameShape(QFrame.StyledPanel)
        self.Header_Left.setFrameShadow(QFrame.Raised)
        self.Header_Left.setLineWidth(0)
        self.Header_Left_Layout = QVBoxLayout(self.Header_Left)
        self.Header_Left_Layout.setSpacing(0)
        self.Header_Left_Layout.setObjectName(u"Header_Left_Layout")
        self.Header_Left_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Left_Signal = QLabel(self.Header_Left)
        self.lb_Left_Signal.setObjectName(u"lb_Left_Signal")
        self.lb_Left_Signal.setLineWidth(0)
        self.lb_Left_Signal.setPixmap(QPixmap(u"images/left_arrow_small.png"))
        self.lb_Left_Signal.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)

        self.Header_Left_Layout.addWidget(self.lb_Left_Signal)


        self.Header_Frame_Layout.addWidget(self.Header_Left)

        self.Header_Title = QFrame(self.Header_Frame)
        self.Header_Title.setObjectName(u"Header_Title")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.Header_Title.sizePolicy().hasHeightForWidth())
        self.Header_Title.setSizePolicy(sizePolicy2)
        self.Header_Title.setFrameShape(QFrame.StyledPanel)
        self.Header_Title.setFrameShadow(QFrame.Raised)
        self.Header_Title.setLineWidth(0)
        self.Header_Title_Layout = QVBoxLayout(self.Header_Title)
        self.Header_Title_Layout.setSpacing(0)
        self.Header_Title_Layout.setObjectName(u"Header_Title_Layout")
        self.Header_Title_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Title = QLabel(self.Header_Title)
        self.lb_Title.setObjectName(u"lb_Title")
        self.lb_Title.setLineWidth(0)
        self.lb_Title.setAlignment(Qt.AlignHCenter|Qt.AlignTop)

        self.Header_Title_Layout.addWidget(self.lb_Title)


        self.Header_Frame_Layout.addWidget(self.Header_Title)

        self.Header_Right = QFrame(self.Header_Frame)
        self.Header_Right.setObjectName(u"Header_Right")
        sizePolicy1.setHeightForWidth(self.Header_Right.sizePolicy().hasHeightForWidth())
        self.Header_Right.setSizePolicy(sizePolicy1)
        self.Header_Right.setFrameShape(QFrame.StyledPanel)
        self.Header_Right.setFrameShadow(QFrame.Raised)
        self.Header_Right.setLineWidth(0)
        self.Header_Right_Layout = QVBoxLayout(self.Header_Right)
        self.Header_Right_Layout.setSpacing(0)
        self.Header_Right_Layout.setObjectName(u"Header_Right_Layout")
        self.Header_Right_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Right_Signal = QLabel(self.Header_Right)
        self.lb_Right_Signal.setObjectName(u"lb_Right_Signal")
        self.lb_Right_Signal.setLineWidth(0)
        self.lb_Right_Signal.setPixmap(QPixmap(u"images/right_arrow_small.png"))
        self.lb_Right_Signal.setAlignment(Qt.AlignRight|Qt.AlignTop|Qt.AlignTrailing)

        self.Header_Right_Layout.addWidget(self.lb_Right_Signal)


        self.Header_Frame_Layout.addWidget(self.Header_Right)

        self.WebviewFrame = QFrame(self.centralwidget)
        self.WebviewFrame.setObjectName(u"WebviewFrame")
        self.WebviewFrame.setGeometry(QRect(0, int(self.width/4+80), 1920, int(self.height-80-self.width/4)))
        sizePolicy1.setHeightForWidth(self.WebviewFrame.sizePolicy().hasHeightForWidth())
        self.WebviewFrame.setSizePolicy(sizePolicy1)
        self.WebviewFrame.setFrameShape(QFrame.StyledPanel)
        self.WebviewFrame.setFrameShadow(QFrame.Raised)
        self.WebviewFrame.setLineWidth(0)
        self.Webview_Layout = QVBoxLayout(self.WebviewFrame)
        self.Webview_Layout.setSpacing(0)
        self.Webview_Layout.setObjectName(u"Webview_Layout")
        self.Webview_Layout.setContentsMargins(0, 0, 0, 0)
        self.WebViewContainer = QWidget(self.WebviewFrame)
        self.WebViewContainer.setObjectName(u"WebViewContainer")
        self.lb_Web = QLabel(self.WebViewContainer)
        self.WebView = QWebEngineView(self.WebViewContainer)
        self.WebView.setObjectName(u"WebView")
        self.WebView.setGeometry(QRect(3, 3, int(self.width-8), int(self.height-80-self.width/4-8)))

        self.Webview_Layout.addWidget(self.WebViewContainer)

        self.BodyFrame = QFrame(self.centralwidget)
        self.BodyFrame.setObjectName(u"BodyFrame")
        self.BodyFrame.setGeometry(QRect(0, 80, self.width, int(self.width/4)))
        sizePolicy1.setHeightForWidth(self.BodyFrame.sizePolicy().hasHeightForWidth())
        self.BodyFrame.setSizePolicy(sizePolicy1)
        self.BodyFrame.setFrameShape(QFrame.StyledPanel)
        self.BodyFrame.setFrameShadow(QFrame.Raised)
        self.BodyFrame.setLineWidth(0)
        self.Body_Left_Frame = QFrame(self.BodyFrame)
        self.Body_Left_Frame.setObjectName(u"Body_Left_Frame")
        self.Body_Left_Frame.setGeometry(QRect(0, 0, int(self.width/4), int(self.width/4)))
        self.Body_Left_Frame.setFrameShape(QFrame.StyledPanel)
        self.Body_Left_Frame.setFrameShadow(QFrame.Raised)
        self.Body_Left_Frame.setLineWidth(0)
        self.verticalLayout_3 = QVBoxLayout(self.Body_Left_Frame)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.lb_Raw_Img = QLabel(self.Body_Left_Frame)
        self.lb_Raw_Img.setObjectName(u"lb_Raw_Img")
        self.lb_Raw_Img.setLineWidth(0)
        self.lb_Raw_Img.setAlignment(Qt.AlignCenter)

        self.verticalLayout_3.addWidget(self.lb_Raw_Img)

        self.Body_Center_Frame = QFrame(self.BodyFrame)
        self.Body_Center_Frame.setObjectName(u"Body_Center_Frame")
        self.Body_Center_Frame.setGeometry(QRect(int(self.width/4), 0, int(self.width/2), int(self.width/4)))
        self.Body_Center_Frame.setFrameShape(QFrame.StyledPanel)
        self.Body_Center_Frame.setFrameShadow(QFrame.Raised)
        self.Body_Center_Frame.setLineWidth(0)
        self.SpeedFrame = QFrame(self.Body_Center_Frame)
        self.SpeedFrame.setObjectName(u"SpeedFrame")
        self.SpeedFrame.setGeometry(QRect(0, 0, int(self.width/4), int(self.width/4)))
        self.SpeedFrame.setFrameShape(QFrame.StyledPanel)
        self.SpeedFrame.setFrameShadow(QFrame.Raised)
        self.Analog_Gauge_Speed = AnalogGaugeWidget(self.SpeedFrame)
        self.Analog_Gauge_Speed.setObjectName(u"Analog_Gauge_Speed")
        self.Analog_Gauge_Speed.setGeometry(QRect(0, 0, int(self.width/4), int(self.width/4)))
        self.lb_Speed = QLabel(self.Analog_Gauge_Speed)
        self.lb_Speed.setObjectName(u"lb_Speed")
        self.lb_Speed.setGeometry(QRect(0, int(self.width/4-40), int(self.width/4), 30))
        self.lb_Speed.setAlignment(Qt.AlignCenter)
        self.AngleFrame = QFrame(self.Body_Center_Frame)
        self.AngleFrame.setObjectName(u"AngleFrame")
        self.AngleFrame.setGeometry(QRect(int(self.width/4), 0, int(self.width/4), int(self.width/4)))
        self.AngleFrame.setFrameShape(QFrame.StyledPanel)
        self.AngleFrame.setFrameShadow(QFrame.Raised)
        self.Analog_Gauge_Angle = AnalogGaugeWidget(self.AngleFrame)
        self.Analog_Gauge_Angle.setObjectName(u"Analog_Gauge_Angle")
        self.Analog_Gauge_Angle.setGeometry(QRect(0, 0, int(self.width/4), int(self.width/4)))
        self.lb_Angle = QLabel(self.Analog_Gauge_Angle)
        self.lb_Angle.setObjectName(u"lb_Angle")
        self.lb_Angle.setGeometry(QRect(0, int(self.width/4-40), int(self.width/4), 30))
        self.lb_Angle.setAlignment(Qt.AlignCenter)
        self.Body_Right_Frame = QFrame(self.BodyFrame)
        self.Body_Right_Frame.setObjectName(u"Body_Right_Frame")
        self.Body_Right_Frame.setGeometry(QRect(int(self.width*3/4), 0, int(self.width/4), int(self.width/4)))
        self.Body_Right_Frame.setFrameShape(QFrame.StyledPanel)
        self.Body_Right_Frame.setFrameShadow(QFrame.Raised)
        self.Body_Right_Frame.setLineWidth(0)
        self.verticalLayout = QVBoxLayout(self.Body_Right_Frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.lb_Output_Img = QLabel(self.Body_Right_Frame)
        self.lb_Output_Img.setObjectName(u"lb_Output_Img")
        self.lb_Output_Img.setLineWidth(0)
        self.lb_Output_Img.setAlignment(Qt.AlignCenter)

        self.verticalLayout.addWidget(self.lb_Output_Img)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"UIT Car Dashboard", None))
        self.lb_Left_Signal.setText("")
        self.lb_Title.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:36pt; font-weight:600; color:#0000ff;\">UIT CAR DASHBOARD</span></p></body></html>", None))
        self.lb_Right_Signal.setText("")
        self.lb_Raw_Img.setText(QCoreApplication.translate("MainWindow", u"Raw_Img", None))
        self.lb_Angle.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:600; color:#00aa00;\">STEERING ANGLE</span></p></body></html>", None))
        self.lb_Speed.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:14pt; font-weight:600; color:#aa5500;\">SPEED</span></p></body></html>", None))
        self.lb_Output_Img.setText(QCoreApplication.translate("MainWindow", u"Output_Img", None))
    # retranslateUi

