"""
Json file imports
"""
import os
import json
import numpy as np

class JsonHandler:
    """
    Class for handling the json
    """
    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), "face_data.json")
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

    def update_json(self,update_value, key=None):
        """
        Method to update the json file
        """
        # Convert numpy types to Python native types
        update_value = self.convert_numpy_types(update_value)
        
        if isinstance(self.data, dict):
            for key in self.data:
                if key == "shapes":
                    self.data[key].extend(update_value["shapes"])
                # if key == "points":
                #     self.data[key] = update_value["points"]
                # if key == "rectangle":
                #     self.data[key] = update_value["rectangle"]
                # if key == "Focus score":
                #     self.data[key] = update_value["focus_score"]
                # elif isinstance(self.data, list):
                #     for item in self.data:
                #         self.update_json(update_value, item)
        print("Unable to edit the json.")
        with open("updated_output.json", "w") as file:
            json.dump(self.data, file, indent=3)
