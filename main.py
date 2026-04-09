import sys
import cv2
import os
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QApplication, QSizePolicy, QHBoxLayout


class AnalyzeThread(QThread):
    # Signal used to send processed image back to the main GUI thread
    resultSignal = pyqtSignal(object)

    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath

        # Load pre-trained Haar Cascade model for car detection
        self.carCascade = cv2.CascadeClassifier('haarcascade_cars.xml')

    def run(self):
        # Ensure cascade file was loaded correctly
        if self.carCascade.empty():
            print("Cascade not loaded!")

        #Load image from file
        img = cv2.imread(self.imagePath)

        # Check if image was loaded successfully
        if img is None:
            print("Image not loaded")
            return

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect cars in the image
        cars = self.carCascade.detectMultiScale(gray,
                                           scaleFactor=1.05,
                                           minNeighbors=5,
                                                minSize=(30, 30))

        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Send processed image back to GUI
        self.resultSignal.emit(img)


class App(QWidget):
    def __init__(self):
        super().__init__()

        self.thread = None
        self.imagePath = None

        # Main horizontal layout that splits UI into image area and control panel
        self.mainLayout = QHBoxLayout()

        # QLabel used as the main image display area
        self.imageLabel = QLabel("Image")
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imageLabel.setMinimumSize(600, 400)

        # Right-side vertical layout for control buttons
        self.rightLayout = QVBoxLayout()

        # Button to load image from filesystem
        self.btnLoadImage = QPushButton("Load Image")
        self.btnLoadImage.clicked.connect(self.loadImage)

        # Button to start analysis
        self.btnAnalyze = QPushButton("Analyze")
        self.btnAnalyze.clicked.connect(self.analyzeImg)

        # Button to clear current image
        self.btnClear = QPushButton("Clear")
        self.btnClear.clicked.connect(self.clearImg)

        # Add flexible spacing so buttons are centered vertically
        self.rightLayout.addStretch()

        # Add control buttons to the right panel
        self.rightLayout.addWidget(self.btnLoadImage)
        self.rightLayout.addWidget(self.btnAnalyze)
        self.rightLayout.addWidget(self.btnClear)

        # Add bottom spacing for better visual balance
        self.rightLayout.addStretch()

        # Add widgets to the main layout with proportional space distribution
        self.mainLayout.addWidget(self.imageLabel, stretch=4)
        self.mainLayout.addLayout(self.rightLayout, stretch=1)

        # Set main layout for the widget
        self.setLayout(self.mainLayout)

    def loadImage(self):
        # Open file dialog and get image path
        imagePath, _ = QFileDialog.getOpenFileName()

        if imagePath:
            self.imagePath = imagePath
            img = cv2.imread(imagePath)
            self.showImage(img)


    def analyzeImg(self):
        # Validate that image is selected
        if not self.imagePath:
            self.imageLabel.setText("Upload a photo")
            return

        # Check if file still exists
        if not os.path.exists(self.imagePath):
            self.imageLabel.setText("File does not exist")
            return

        # Stop previous thread if it's still running
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        # Start new analysis thread
        self.thread = AnalyzeThread(self.imagePath)
        self.thread.resultSignal.connect(self.showImage)
        self.thread.start()


    def showImage(self, img):
        # Convert OpenCV BGR image to RGB format for Qt
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w

        # Convert numpy image to QImage
        qtImage = QImage(
            rgbImage.data.tobytes(),
            w, h,
            bytesPerLine,
            QImage.Format.Format_RGB888
        )

        # Display image in QLabel with proper scaling
        self.imageLabel.setPixmap(QPixmap.fromImage(qtImage).scaled(
            self.imageLabel.width(),
            self.imageLabel.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))


    def clearImg(self):
        # Stop running thread if needed
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        # Reset UI state
        self.imageLabel.clear()
        self.imagePath = None

app = QApplication(sys.argv)

window = App()
window.show()

app.exec()