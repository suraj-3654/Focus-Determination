"""
Imports for the annotation tool.
"""
import os
import sys
import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QHBoxLayout
)
from media_pipe.segment_facial_parts import SegmentFacialParts

class AnnotationViewer(QWidget):
    """
    This class contatins the functions for annotation tool.
    """

    # Creating custom stylesheet for QPushButton
    button_style = """
    QPushButton {
        background-color: red;
        color: white;
        font-size: 16px;
        padding: 10px;
    }
    QPushButton:hover {
        background-color: darkred;
        }
    """
    def __init__(self):
        """contructor"""
        super().__init__()
        self.setWindowTitle("Annotation Tool Viewer.")
        self.setGeometry(250, 100, 1500, 900)
        self.setStyleSheet("background-color: #e0f7fa;")
        self.input_folder_path = None
        self.file_list = []

        # Creating custom stylesheet for QPushButton
        button_style = """
        QPushButton {
            background-color: #3498db;       /* Blue background */
            color: white;                    /* White text */
            font-size: 16px;                 /* Font size */
            padding: 10px 20px;              /* Vertical and horizontal padding */
            border: none;                    /* No border */
            border-radius: 8px;              /* Rounded corners */
            font-weight: bold;              /* Optional: bold text */
        }

        QPushButton:hover {
            background-color: #2980b9;       /* Darker blue on hover */
        }

        QPushButton:pressed {
            background-color: #1c6690;       /* Even darker when pressed */
        }

        QPushButton:disabled {
            background-color: #bdc3c7;       /* Gray when disabled */
            color: #7f8c8d;
        }
        """

        self.select_folder = QPushButton("Select folder")
        self.select_folder.clicked.connect(self.load_folder)

        self.clear_folder = QPushButton("Clear Folder Data")
        self.clear_folder.clicked.connect(self.clear_folder_data)

        self.start_annotation_button = QPushButton("start annotation")
        self.start_annotation_button.clicked.connect(self.start_annotation)
        self.select_folder.setStyleSheet(button_style)
        self.clear_folder.setStyleSheet(button_style)
        self.start_annotation_button.setStyleSheet(button_style)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.select_folder)
        top_bar.addWidget(self.clear_folder)
        top_bar.addWidget(self.start_annotation_button)
        self.start_annotation_button.setEnabled(False)

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
            self.start_annotation_button.setEnabled(True)
            # print("Files loaded:", self.file_list)

    def clear_folder_data(self):
        """
        This method is to clear the selected folder
        """
        if self.input_folder_path:
            self.input_folder_path = None
            self.file_list = []
            self.start_annotation_button.setEnabled(False)
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
