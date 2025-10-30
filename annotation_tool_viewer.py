"""
Imports for the annotation tool.
"""
import os
import sys

# === FIX MEDIAPIPE DLL LOAD IN PYINSTALLER/NUITKA EXE ===
if getattr(sys, 'frozen', False):
    # Running as .exe (bundled by PyInstaller or Nuitka)
    # PyInstaller uses sys._MEIPASS, Nuitka uses os.path.dirname(sys.executable)
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS  # PyInstaller
    else:
        base_path = os.path.dirname(sys.executable)  # Nuitka
    
    # AGGRESSIVE DLL path setup - add ALL directories containing DLLs
    dll_dirs = []
    
    # Add base and common paths
    dll_dirs.append(base_path)
    dll_dirs.append(os.path.join(base_path, 'mediapipe'))
    dll_dirs.append(os.path.join(base_path, 'mediapipe', 'python'))
    dll_dirs.append(os.path.join(base_path, 'cv2'))
    dll_dirs.append(os.path.join(base_path, 'numpy', '.libs'))
    dll_dirs.append(os.path.join(base_path, 'PyQt5', 'Qt5', 'bin'))
    
    # Walk and find all DLL directories (up to 3 levels deep)
    for root, dirs, files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep)
        if depth <= 3:
            # Check if this directory has DLLs
            if any(f.lower().endswith(('.dll', '.pyd')) for f in files):
                if root not in dll_dirs:
                    dll_dirs.append(root)
    
    # Add ALL to DLL search paths
    path_parts = []
    for d in dll_dirs:
        if os.path.exists(d):
            os.add_dll_directory(d)
            path_parts.append(d)
    
    # Set PATH to include all DLL directories
    os.environ['PATH'] = os.pathsep.join(path_parts) + os.pathsep + os.environ.get('PATH', '')

import mediapipe as mp

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QHBoxLayout,
    QVBoxLayout, QLabel, QTextEdit, QMessageBox, QProgressBar, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from media_pipe.segment_facial_parts import SegmentFacialParts

class ProcessingThread(QThread):
    """Thread for processing images without freezing the GUI"""
    finished = pyqtSignal(str, bool)  # message, success
    status_update = pyqtSignal(str)  # status message
    
    def __init__(self, segment_obj, folder_path):
        super().__init__()
        self.segment_obj = segment_obj
        self.folder_path = folder_path
    
    def run(self):
        """Run the processing in background thread"""
        import sys
        from io import StringIO
        
        # Redirect stdout/stderr to capture print statements
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_output = StringIO()
        sys.stdout = captured_output
        sys.stderr = captured_output
        
        try:
            self.status_update.emit(f">>> Starting processing of folder: {self.folder_path}")
            self.status_update.emit("=" * 70)
            
            # Process the folder
            self.segment_obj.process_folder(self.folder_path)
            
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Get captured output and emit as status updates
            output_text = captured_output.getvalue()
            if output_text:
                for line in output_text.split('\n'):
                    if line.strip():
                        self.status_update.emit(line)
            
            # Get output directory
            if self.segment_obj.output_dir and os.path.exists(self.segment_obj.output_dir):
                output_msg = f"Processing complete!\nOutput saved to:\n{self.segment_obj.output_dir}"
                self.finished.emit(output_msg, True)
            else:
                self.finished.emit("Processing completed but output directory not found. Check for errors.", False)
                
        except Exception as e:
            # Restore stdout/stderr in case of error
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Also get any captured output before the error
            output_text = captured_output.getvalue()
            if output_text:
                for line in output_text.split('\n'):
                    if line.strip():
                        self.status_update.emit(line)
            
            error_msg = f"Error during processing:\n{str(e)}\n\nDetails: {type(e).__name__}"
            import traceback
            error_msg += f"\n\n{traceback.format_exc()}"
            self.finished.emit(error_msg, False)

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
        self.setWindowTitle("Annotation Tool Viewer")
        self.setGeometry(250, 100, 1500, 900)
        self.setStyleSheet("background-color: #e0f7fa;")
        self.input_folder_path = None
        self.file_list = []
        self.obj_segment_factil_part = SegmentFacialParts.get_instance()
        self.processing_thread = None

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
        top_bar.addWidget(self.start_annotation_button)
        top_bar.addWidget(self.clear_folder)
        self.start_annotation_button.setEnabled(False)

        # Add status label
        self.status_label = QLabel("Status: Ready - Select a folder to begin")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border: 2px solid #3498db;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.status_label.setWordWrap(True)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #3498db;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        
        # Output folder display
        output_folder_label = QLabel("Output Folder:")
        output_folder_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        self.output_folder_display = QLineEdit()
        self.output_folder_display.setReadOnly(True)
        self.output_folder_display.setPlaceholderText("Output folder will appear here after processing...")
        self.output_folder_display.setStyleSheet("""
            QLineEdit {
                background-color: #f8f9fa;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        
        output_folder_layout = QHBoxLayout()
        output_folder_layout.addWidget(output_folder_label)
        output_folder_layout.addWidget(self.output_folder_display)
        
        # Add terminal-like output area (MAIN FEATURE)
        terminal_label = QLabel("Processing Output (Terminal View):")
        terminal_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 2px solid #3498db;
                border-radius: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 10px;
            }
        """)
        # Make terminal output take most of the space
        self.terminal_output.setMinimumHeight(400)
        
        # Initial welcome message
        welcome_msg = "=" * 70 + "\n"
        welcome_msg += "Annotation Tool Viewer - Terminal Output\n"
        welcome_msg += "=" * 70 + "\n"
        welcome_msg += "Ready. Please select a folder and click 'start annotation' to begin.\n"
        welcome_msg += "=" * 70 + "\n"
        self.terminal_output.append(welcome_msg)
        
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_bar)
        main_layout.addSpacing(10)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.progress_bar)
        main_layout.addSpacing(10)
        main_layout.addLayout(output_folder_layout)
        main_layout.addSpacing(5)
        main_layout.addWidget(terminal_label)
        main_layout.addWidget(self.terminal_output)
        
        self.setLayout(main_layout)
        
        # Verify model file exists (must be after UI elements are created)
        self._verify_model_file()
    
    def _verify_model_file(self):
        """Verify that the model file exists and can be accessed"""
        try:
            from utility.resource_path import resource_path
            model_path = resource_path("model/focusmodel.onnx")
            if os.path.exists(model_path):
                pass  # Model file verified
            else:
                self.log_terminal(f"WARNING: Model file not found at: {model_path}")
                QMessageBox.warning(self, "Model File Missing", 
                                  f"Model file not found!\n\nExpected: {model_path}\n\n"
                                  "Please ensure the model file is included in the build.")
        except Exception as e:
            self.log_terminal(f"Error checking model file: {str(e)}")
            QMessageBox.warning(self, "Model Check Error", f"Could not verify model file:\n{str(e)}")
    
    def log_terminal(self, message):
        """Add a message to the terminal output"""
        self.terminal_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.terminal_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def load_folder(self):
        """
        This folder is to select the image folder
        """
        self.input_folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.input_folder_path:
            self.log_terminal(f">>> Selected folder: {self.input_folder_path}")
            self.status_label.setText(f"Status: Folder selected - {os.path.basename(self.input_folder_path)}")
            self.start_annotation_button.setEnabled(True)
            # Clear output folder display when new folder is selected
            self.output_folder_display.clear()

    def clear_folder_data(self):
        """
        This method is to clear the selected folder
        """
        if self.input_folder_path:
            self.input_folder_path = None
            self.file_list = []
            self.start_annotation_button.setEnabled(False)
            self.obj_segment_factil_part.clear_data()
            self.log_terminal(">>> Folder contents cleared.")
            self.status_label.setText("Status: Ready - Select a folder to begin")
            self.output_folder_display.clear()

    def start_annotation(self):
        """
        function to start the annotation with selected images
        """
        if not self.input_folder_path:
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder first!")
            return
        
        if not os.path.exists(self.input_folder_path):
            QMessageBox.critical(self, "Folder Not Found", 
                               f"The selected folder no longer exists:\n{self.input_folder_path}")
            return
        
        # Clear terminal and show start message
        self.terminal_output.clear()
        self.log_terminal("=" * 70)
        self.log_terminal("STARTING ANNOTATION PROCESSING")
        self.log_terminal("=" * 70)
        self.log_terminal(f"Input Folder: {self.input_folder_path}")
        self.log_terminal("-" * 70)
        
        # Disable button during processing
        self.start_annotation_button.setEnabled(False)
        self.select_folder.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Status: Processing images... Please wait.")
        
        # Clear previous output folder
        self.output_folder_display.clear()
        
        # Create and start processing thread
        self.processing_thread = ProcessingThread(self.obj_segment_factil_part, self.input_folder_path)
        self.processing_thread.status_update.connect(self.log_terminal)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.start()
    
    def on_processing_finished(self, message, success):
        """Handle processing completion"""
        self.progress_bar.setVisible(False)
        self.start_annotation_button.setEnabled(True)
        self.select_folder.setEnabled(True)
        
        self.log_terminal("-" * 70)
        
        if success:
            self.status_label.setText(f"Status: Processing complete")
            self.log_terminal("=" * 70)
            self.log_terminal("PROCESSING COMPLETED SUCCESSFULLY")
            self.log_terminal("=" * 70)
            
            # Update output folder display
            if self.obj_segment_factil_part.output_dir:
                self.output_folder_display.setText(self.obj_segment_factil_part.output_dir)
                self.log_terminal(f"Output Directory: {self.obj_segment_factil_part.output_dir}")
            
            QMessageBox.information(self, "Processing Complete", message)
        else:
            self.status_label.setText(f"Status: Processing failed")
            self.log_terminal("=" * 70)
            self.log_terminal("PROCESSING FAILED")
            self.log_terminal("=" * 70)
            QMessageBox.critical(self, "Processing Error", message)
        
        self.log_terminal("=" * 70)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnnotationViewer()
    window.show()
    sys.exit(app.exec_())
