'''
date: 2020/01/02
author: wqx
'''
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from detection import run_detection
from PIL import Image, ImageQt


class imageProcessing(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Detect Caps"
        self.left = 10
        self.top = 10
        self.width = 1000
        self.height = 540
        self.ksize = 3
        self.sigma = 1.0
        self.cur_image = np.ndarray(())
        self.raw_imgs=[]
        self.raw_img_files = []
        self.detect_imgs = []
        self.page = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        selectLabel = QLabel('Target Directory: ')
        selectBtn = QPushButton("Select Images")
        detectBtn = QPushButton("Detect All")
        # saveBtn = QPushButton("Save")
        cancelBtn = QPushButton("Quit")
        self.imgLabel_raw = QLabel()
        self.imgLabel_detect = QLabel()

        prevBtn = QPushButton("< Prev")
        nextBtn = QPushButton("Next >")

        # connection
        detectBtn.clicked.connect(self.detect)
        selectBtn.clicked.connect(self.load_image)
        cancelBtn.clicked.connect(self.close)
        prevBtn.clicked.connect(self.prev_image)
        nextBtn.clicked.connect(self.next_image)

        groupbox1 = QGroupBox('Operation', self)
        hboxLeftUp = QHBoxLayout()
        hboxLeftUp.addWidget(selectLabel, 0, Qt.AlignLeft)
        hboxLeftUp.addWidget(selectBtn, 0, Qt.AlignLeft)
        hboxLeftUp.addWidget(detectBtn, 0, Qt.AlignLeft)
        hboxLeftUp.addStretch(5)
        hboxLeftUp.addWidget(cancelBtn, 0, Qt.AlignLeft)

        groupbox1.setLayout(hboxLeftUp)

        groupbox2 = QGroupBox('Display', self)
        vboxMid = QVBoxLayout()
        hboxMid = QHBoxLayout()
        hboxMid.addWidget(self.imgLabel_raw)
        hboxMid.addWidget(self.imgLabel_detect)
        hboxMidDown = QHBoxLayout()
        hboxMidDown.addStretch(10)
        hboxMidDown.addWidget(prevBtn, Qt.AlignHCenter)
        hboxMidDown.addWidget(nextBtn, Qt.AlignHCenter)
        hboxMidDown.addStretch(10)
        vboxMid.addLayout(hboxMid)
        vboxMid.addLayout(hboxMidDown)
        groupbox2.setLayout(vboxMid)

        vboxAll = QVBoxLayout()
        vboxAll.addWidget(groupbox1)
        vboxAll.addStretch(1)
        vboxAll.addWidget(groupbox2)

        self.setLayout(vboxAll)
        self.show()

    def prev_image(self):
        if self.page > 0:
            self.page -= 1
        self.show_image()

    def next_image(self):
        if self.page < len(self.raw_imgs)-1:
            self.page += 1
        self.show_image()

    def show_image(self):
        # show image
        # im = Image.open(self.raw_imgs[self.page])
        # im = im.convert("RGBA")
        # im.thumbnail((450, 450), Image.ANTIALIAS)
        if not self.raw_imgs:
            QMessageBox.information(self, 'Fail', 'No image selected.')
            return
        im = self.raw_imgs[self.page]
        img = ImageQt.ImageQt(im)
        qimg = QPixmap.fromImage(img)
        self.imgLabel_raw.setPixmap(qimg)
        self.imgLabel_raw.resize(1, 1)

        # show labeld image
        if len(self.detect_imgs) > self.page:
            self.labeled_img = self.detect_imgs[self.page]
            # convert PIL to Qimage
            img = ImageQt.ImageQt(self.labeled_img)
            img = QPixmap.fromImage(img)
            self.imgLabel_detect.setPixmap(img)
            self.imgLabel_detect.resize(1, 1)
        if not self.detect_imgs:
            self.imgLabel_detect.clear()

    def detect(self):
        self.detect_imgs = []
        for file in self.raw_img_files:
            im2 = run_detection(file)
            im2 = im2.convert("RGBA")
            im2.thumbnail((450, 450), Image.ANTIALIAS)
            self.detect_imgs.append(im2)
        self.show_image()

    def load_image(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'select images', './', 'Image files(*.jpg)')
        if not files:
            QMessageBox.information(self, 'Fail', 'No image selected.')
            return
        self.raw_img_files = files
        self.raw_imgs = []
        self.detect_imgs = []
        for file in files:
            im = Image.open(file)
            im = im.convert("RGBA")
            im.thumbnail((450, 450), Image.ANTIALIAS)
            self.raw_imgs.append(im)

        print(self.raw_img_files)
        # show image
        self.show_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = imageProcessing()
    sys.exit(app.exec_())
