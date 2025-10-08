"""
Imports for the annotation tool.
"""
import os
import sys
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QScrollArea
)
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QMargins
from media_pipe.adaptive_forehead_extraction import SegmentFacialParts

class AnnotationViewer(QWidget):
    """
    This class contatins the functions for annotation tool.
    """
    def __init__(self):
        """contructor"""
        super().__init__()
        self.setWindowTitle("Annotation Tool Viewer.")
        self.setGeometry(250, 100, 1500, 900)
        self.input_folder_path = None
        self.file_list = []

        self.select_folder = QPushButton("Select folder")
        self.select_folder.clicked.connect(self.load_folder)

        self.clear_folder = QPushButton("Clear Folder Data")
        self.clear_folder.clicked.connect(self.clear_folder_data)

        self.start_annotation_button = QPushButton("start annotation")
        self.start_annotation_button.clicked.connect(self.start_annotation)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.select_folder)
        top_bar.addWidget(self.clear_folder)
        top_bar.addWidget(self.start_annotation_button)

        self.setLayout(top_bar)

    def load_folder(self):
        """
        This folder is to select the image folder
        """
        self.input_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.input_folder_path:
            # for file_name in os.listdir(self.input_folder_path):
            #     fullpath = os.path.join(self.input_folder_path, file_name)
            #     if os.path.isfile(fullpath):
            #         self.file_list.append(fullpath)
            print("Selected folder:", self.input_folder_path)
            # print("Files loaded:", self.file_list)

    def clear_folder_data(self):
        """
        This method is to clear the selected folder
        """
        if self.input_folder_path:
            self.input_folder_path = None
            self.file_list = []
            print("Folder contents cleared.")

    def start_annotation(self):
        """
        function to start the annotation with selected images
        """
        obj_segment_facial_parts = SegmentFacialParts()
        obj_segment_facial_parts.process_folder(self.input_folder_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationViewer()
    window.show()
    sys.exit(app.exec_())
