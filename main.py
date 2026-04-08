import sys
import cv2
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QApplication, QSizePolicy


class AnalyzeThread(QThread):
    resultSignal = pyqtSignal(object)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')
        img = cv2.imread(self.path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.01, 5)

        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.resultSignal.emit(img)


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.thread = None
        self.layout = QVBoxLayout()

        self.label = QLabel("Image")
        self.label.setMinimumSize(600, 400)

        self.btnLoadImage = QPushButton("Load Image")
        self.btnLoadImage.clicked.connect(self.loadImage)
        self.btnLoadImage.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.btnAnalyze = QPushButton("Analyze")
        self.btnAnalyze.clicked.connect(self.analyzeImg)

        self.btnClear = QPushButton("Clear")
        self.btnClear.clicked.connect(self.clearImg)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.btnLoadImage)
        self.layout.addWidget(self.btnAnalyze)
        self.layout.addWidget(self.btnClear)

        self.setLayout(self.layout)

    def loadImage(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.imagePath = path
            img = cv2.imread(path)
            self.showImage(img)


    def analyzeImg(self):
        self.thread = AnalyzeThread(self.imagePath)
        self.thread.resultSignal.connect(self.showImage)
        self.thread.start()


    def showImage(self, img):
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w

        qtImage = QImage(
            rgbImage.data.tobytes(),
            w, h,
            bytesPerLine,
            QImage.Format.Format_RGB888
        )

        self.label.setPixmap(QPixmap.fromImage(qtImage).scaled(
            self.label.width(),
            self.label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))


    def clearImg(self):
        if self.thread and self.thread.isRunning():
            self.thread.quit()
        self.label.clear()
        self.imagePath = None

app = QApplication(sys.argv)

window = App()
window.show()

app.exec()






