"""
Resource path utility for PyInstaller compatibility.
This module provides a function to get the correct resource path
whether running as a script or as a PyInstaller bundle.
"""
import os
import sys


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and PyInstaller.
    
    When running as a PyInstaller bundle, resources are extracted
    to a temporary directory and referenced via sys._MEIPASS.
    
    Args:
        relative_path: Path relative to the application root
        
    Returns:
        str: Absolute path to the resource
    """
    if getattr(sys, 'frozen', False):
        # Running as bundled executable
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller
            base_path = sys._MEIPASS
        else:
            # Nuitka - resources are in same directory as exe
            base_path = os.path.dirname(sys.executable)
    else:
        # Running as a normal Python script
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


