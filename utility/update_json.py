"""
Json file imports
"""
import os
import json
import numpy as np
from utility.resource_path import resource_path

class JsonHandler:
    """
    Class for handling the json
    """
    def __init__(self):
        file_path = resource_path("utility/face_data.json")
        with open(file_path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def convert_numpy_types(self, obj):
        """
        Convert numpy types to Python native types for JSON serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj

    def update_json(self, update_value, output_path=None, image_info=None, key=None):
        """
        Method to update the json file
        Args:
            update_value: Data to add to JSON
            output_path: Custom path to save JSON file (optional)
            image_info: Dictionary containing imagePath, imageHeight, imageWidth (optional)
            key: Key for updating (optional)
        """
        # Convert numpy types to Python native types
        update_value = self.convert_numpy_types(update_value)

        if isinstance(self.data, dict):
            for key in self.data:
                if key == "shapes":
                    self.data[key].extend(update_value["shapes"])
        # Update image metadata if provided
        if image_info:
            if "imagePath" in image_info:
                self.data["imagePath"] = image_info["imagePath"]
            if "imageHeight" in image_info:
                self.data["imageHeight"] = image_info["imageHeight"]
            if "imageWidth" in image_info:
                self.data["imageWidth"] = image_info["imageWidth"]

        # Determine output path
        if output_path:
            json_file_path = os.path.join(output_path, "annotation_data.json")
        else:
            json_file_path = "updated_output.json"

        with open(json_file_path, "w") as file:
            json.dump(self.data, file, indent=3)
