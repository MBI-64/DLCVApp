from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import PyQt5.QtGui as QtGui
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2
import sys
import glob
import os
import numpy as np
import time

class ImageThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, image_paths, fps=1):
        super().__init__()
        self.image_paths = image_paths
        self._run_flag = True
        self.paused = False
        self.current_index = 0
        self.grayscale = False
        self.edge_filter = False
        self.fps = fps

    def run(self):
        while self._run_flag:
            if not self.paused:
                img_path = self.image_paths[self.current_index]
                img = cv2.imread(img_path)
                if img is not None:
                    if self.grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif self.edge_filter:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(img_gray, 100, 200)
                        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.change_pixmap_signal.emit(img)
                    self.current_index = (self.current_index + 1) % len(self.image_paths)
                time.sleep(1 / self.fps)

    def stop(self):
        self._run_flag = False
        self.wait()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_grayscale(self, grayscale):
        self.grayscale = grayscale

    def set_edge_filter(self, edge_filter):
        self.edge_filter = edge_filter

    def set_fps(self, fps):
        self.fps = fps

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, video_path, fps):
        super().__init__()
        self.video_path = video_path
        self.fps = fps
        self._run_flag = True
        self.paused = False
        self.grayscale = False
        self.edge_filter = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return

        while self._run_flag and self.cap.isOpened():
            if not self.paused:
                ret, img = self.cap.read()
                if ret:
                    if self.grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif self.edge_filter:
                        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(img_gray, 100, 200)
                        img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.change_pixmap_signal.emit(img)
                    time.sleep(1 / self.fps)
                else:
                    break

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_grayscale(self, grayscale):
        self.grayscale = grayscale

    def set_edge_filter(self, edge_filter):
        self.edge_filter = edge_filter

    def set_fps(self, fps):
        self.fps = fps

class CamThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, fps=30):
        super().__init__()
        self._run_flag = True
        self.paused = False
        self.grayscale = False
        self.edge_filter = False
        self.fps = fps

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open the webcam")
            return

        while self._run_flag:
            if not self.paused:
                ret, cv_img = cap.read()
                if ret:
                    if self.grayscale:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
                    elif self.edge_filter:
                        img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(img_gray, 100, 200)
                        cv_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                    else:
                        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    self.change_pixmap_signal.emit(cv_img)
                    time.sleep(1 / self.fps)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def set_grayscale(self, grayscale):
        self.grayscale = grayscale

    def set_edge_filter(self, edge_filter):
        self.edge_filter = edge_filter

    def set_fps(self, fps):
        self.fps = fps

class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        uic.loadUi("camtest.ui", self)

        self.pushButtonStart.clicked.connect(self.startAcq)
        self.pushButtonStop.clicked.connect(self.stopAcq)

        self.actionImage_Folder.triggered.connect(self.newImgPressed)
        self.actionOffline_Video.triggered.connect(self.newVidPressed)
        self.actionLive_Cam_Stream.triggered.connect(self.newCamPressed)

        self.action1_FPS.triggered.connect(lambda: self.change_fps(1))
        self.action3_FPS.triggered.connect(lambda: self.change_fps(3))
        self.action10_FPS.triggered.connect(lambda: self.change_fps(10))
        self.action25_FPS.triggered.connect(lambda: self.change_fps(25))

        self.actionGrayScale.triggered.connect(self.grayscalePressed)
        self.actionEdge_Filter.triggered.connect(self.edgeFilterPressed)

        self.thread = None

        self.imgBox.setScaledContents(True)
        self.show()

    def startAcq(self):
        if self.thread is not None:
            self.thread.resume()
        else:
            self.imgBox.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
            self.thread = CamThread()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.start()

    @pyqtSlot(np.ndarray)
    def update_image(self, image_np):
        img = Image.fromarray(image_np)
        qim = ImageQt(img)
        pixmap = QtGui.QPixmap.fromImage(qim)
        self.imgBox.setPixmap(pixmap.scaled(self.imgBox.size()))

    def stopAcq(self):
        if isinstance(self.thread, (ImageThread, VideoThread, CamThread)):
            self.thread.pause()

    def newImgPressed(self):
        print("Image Folder was pressed")
        folder_path = QFileDialog.getExistingDirectory()
        if folder_path:
            image_paths = sorted(glob.glob(os.path.join(folder_path, '*.jpg')) + 
                                 glob.glob(os.path.join(folder_path, '*.jpeg')) +
                                 glob.glob(os.path.join(folder_path, '*.png')))
            if image_paths:
                self.stopAcq()
                self.thread = ImageThread(image_paths, fps=1)  
                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.start()

    def newVidPressed(self):
        print("Select Video File")
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if video_path:
            self.start_video_thread(video_path, 26)

    def newCamPressed(self):
        print("Web Cam Active")
        self.stopAcq()
        self.thread = CamThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def start_video_thread(self, video_path, fps):
        self.stopAcq()  
        self.thread = VideoThread(video_path, fps)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def change_fps(self, fps):
        if isinstance(self.thread, (ImageThread, VideoThread, CamThread)):
            self.thread.set_fps(fps)

    def grayscalePressed(self):
        print("GrayScale Active")
        if isinstance(self.thread, (ImageThread, VideoThread, CamThread)):
            self.thread.set_grayscale(True)

    def edgeFilterPressed(self):
        print("Edge Filter Active")
        if isinstance(self.thread, (ImageThread, VideoThread, CamThread)):
            self.thread.set_edge_filter(True)

    def closeEvent(self, event):
        self.stopAcq()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainUI()
    ui.show()
    sys.exit(app.exec_())



