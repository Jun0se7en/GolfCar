# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindowtHRnfH.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QSizePolicy, QVBoxLayout,
    QWidget)

from Custom_Widgets.AnalogGaugeWidget import AnalogGaugeWidget

from PySide6.QtWebEngineWidgets import QWebEngineView

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1278, 720)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.Header_Frame = QFrame(self.centralwidget)
        self.Header_Frame.setObjectName(u"Header_Frame")
        self.Header_Frame.setGeometry(QRect(0, 0, 1281, 81))
        self.Header_Frame.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.Header_Frame.setStyleSheet(u"")
        self.Header_Frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.Header_Frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout = QHBoxLayout(self.Header_Frame)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.Header_Left = QFrame(self.Header_Frame)
        self.Header_Left.setObjectName(u"Header_Left")
        self.Header_Left.setStyleSheet(u"border:0")
        self.Header_Left.setFrameShape(QFrame.Shape.StyledPanel)
        self.Header_Left.setFrameShadow(QFrame.Shadow.Raised)
        self.Header_Left.setLineWidth(0)
        self.verticalLayout_6 = QVBoxLayout(self.Header_Left)
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.lb_Left_Signal = QLabel(self.Header_Left)
        self.lb_Left_Signal.setObjectName(u"lb_Left_Signal")
        self.lb_Left_Signal.setPixmap(QPixmap(u"images/left_arrow_small.png"))

        self.verticalLayout_6.addWidget(self.lb_Left_Signal, 0, Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)


        self.horizontalLayout.addWidget(self.Header_Left)

        self.Header_Title = QFrame(self.Header_Frame)
        self.Header_Title.setObjectName(u"Header_Title")
        self.Header_Title.setStyleSheet(u"border:0")
        self.Header_Title.setFrameShape(QFrame.Shape.StyledPanel)
        self.Header_Title.setFrameShadow(QFrame.Shadow.Raised)
        self.Header_Title.setLineWidth(0)
        self.verticalLayout_4 = QVBoxLayout(self.Header_Title)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.lb_Title = QLabel(self.Header_Title)
        self.lb_Title.setObjectName(u"lb_Title")
        font = QFont()
        font.setPointSize(40)
        font.setBold(True)
        self.lb_Title.setFont(font)
        self.lb_Title.setStyleSheet(u"border:0")
        self.lb_Title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_4.addWidget(self.lb_Title, 0, Qt.AlignmentFlag.AlignVCenter)


        self.horizontalLayout.addWidget(self.Header_Title, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)

        self.Header_Right = QFrame(self.Header_Frame)
        self.Header_Right.setObjectName(u"Header_Right")
        self.Header_Right.setStyleSheet(u"border:0")
        self.Header_Right.setFrameShape(QFrame.Shape.StyledPanel)
        self.Header_Right.setFrameShadow(QFrame.Shadow.Raised)
        self.Header_Right.setLineWidth(0)
        self.verticalLayout_5 = QVBoxLayout(self.Header_Right)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.lb_Right_Signal = QLabel(self.Header_Right)
        self.lb_Right_Signal.setObjectName(u"lb_Right_Signal")
        self.lb_Right_Signal.setPixmap(QPixmap(u"images/right_arrow_small.png"))

        self.verticalLayout_5.addWidget(self.lb_Right_Signal, 0, Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)


        self.horizontalLayout.addWidget(self.Header_Right)

        self.Body_Frame = QFrame(self.centralwidget)
        self.Body_Frame.setObjectName(u"Body_Frame")
        self.Body_Frame.setGeometry(QRect(0, 80, 1280, 242))
        self.Body_Frame.setMinimumSize(QSize(121, 0))
        self.Body_Frame.setStyleSheet(u"")
        self.Body_Frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.Body_Frame.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.Body_Frame)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.Body_Left_Frame = QFrame(self.Body_Frame)
        self.Body_Left_Frame.setObjectName(u"Body_Left_Frame")
        self.Body_Left_Frame.setStyleSheet(u"border:0")
        self.Body_Left_Frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.Body_Left_Frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.Body_Left_Frame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.lb_Raw_Img = QLabel(self.Body_Left_Frame)
        self.lb_Raw_Img.setObjectName(u"lb_Raw_Img")

        self.verticalLayout_3.addWidget(self.lb_Raw_Img, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)


        self.horizontalLayout_2.addWidget(self.Body_Left_Frame)

        self.Body_Center_Frame = QFrame(self.Body_Frame)
        self.Body_Center_Frame.setObjectName(u"Body_Center_Frame")
        self.Body_Center_Frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.Body_Center_Frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.Body_Center_Frame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.label_3 = QLabel(self.Body_Center_Frame)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.Analog_Gauge_Speed = AnalogGaugeWidget(self.Body_Center_Frame)
        self.Analog_Gauge_Speed.setObjectName(u"Analog_Gauge_Speed")
        self.Analog_Gauge_Speed.setEnabled(False)
        self.Analog_Gauge_Speed.setMinimumSize(QSize(200, 200))

        self.gridLayout.addWidget(self.Analog_Gauge_Speed, 0, 1, 2, 1, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)

        self.label_4 = QLabel(self.Body_Center_Frame)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1, Qt.AlignmentFlag.AlignRight)

        self.label_5 = QLabel(self.Body_Center_Frame)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.label_6 = QLabel(self.Body_Center_Frame)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout.addWidget(self.label_6, 1, 2, 1, 1, Qt.AlignmentFlag.AlignRight)


        self.horizontalLayout_2.addWidget(self.Body_Center_Frame)

        self.Body_Right_Frame = QFrame(self.Body_Frame)
        self.Body_Right_Frame.setObjectName(u"Body_Right_Frame")
        self.Body_Right_Frame.setStyleSheet(u"border:0")
        self.Body_Right_Frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.Body_Right_Frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.Body_Right_Frame)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.lb_Output_Img = QLabel(self.Body_Right_Frame)
        self.lb_Output_Img.setObjectName(u"lb_Output_Img")

        self.verticalLayout_2.addWidget(self.lb_Output_Img, 0, Qt.AlignmentFlag.AlignHCenter|Qt.AlignmentFlag.AlignVCenter)


        self.horizontalLayout_2.addWidget(self.Body_Right_Frame)

        self.Webview_Frame = QFrame(self.centralwidget)
        self.Webview_Frame.setObjectName(u"Webview_Frame")
        self.Webview_Frame.setGeometry(QRect(0, 322, 1280, 400))
        self.Webview_Frame.setMinimumSize(QSize(1280, 400))
        self.Webview_Frame.setStyleSheet(u"border:0")
        self.Webview_Frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.Webview_Frame.setFrameShadow(QFrame.Shadow.Raised)
        self.verticalLayout = QVBoxLayout(self.Webview_Frame)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.WebViewContainer = QWidget(self.Webview_Frame)
        self.WebViewContainer.setObjectName(u"WebViewContainer")
        self.verticalLayout_7 = QVBoxLayout(self.WebViewContainer)
        self.verticalLayout_7.setSpacing(0)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.WebView = QWebEngineView(self.WebViewContainer)
        self.WebView.setObjectName(u"WebView")

        self.verticalLayout_7.addWidget(self.WebView)


        self.verticalLayout.addWidget(self.WebViewContainer)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"UIT Car Dashboard", None))
        self.lb_Left_Signal.setText("")
        self.lb_Title.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:28pt; color:#0055ff;\">UIT CAR DASHBOARD</span></p></body></html>", None))
        self.lb_Right_Signal.setText("")
        self.lb_Raw_Img.setText(QCoreApplication.translate("MainWindow", u"Raw_Img", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Brake", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Front Light", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Horn", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Rear Light", None))
        self.lb_Output_Img.setText(QCoreApplication.translate("MainWindow", u"Output", None))
    # retranslateUi

