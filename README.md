# Focus-Determination

## Facial Parts Segmentation using MediaPipe
This repository provides a pipeline for segmenting facial regions such as the left eye, right eye, full face, and head using MediaPipe. It supports both single-person and group images.

## 🧰 Environment Setup

> Requirement: Use Python 3.10 (recommended/validated).
### 1. Clone the Repository
```bash
git clone https://github.com/suraj-3654/Focus-Determination.git
cd Focus-Determination
```

### 2. Create a python environment
```bash
python -m venv env
cd env\Scripts\activate
```

### 3. Install packge and libraries
Install the required pakages.
```
pip install -r requirements.txt
```

### 4. start annotation tool
After completing the mentioned steps you can start the annotation tool by,**python annotation_tool_viewer.py**
After launching click on the **select folder** button and click on a folder containing the images.
Then click on the **start annotation** button to facial parts segmentation.

After successful segmentation, the cropped facial images are saved in a folder named **segmented_images**, located next to your input folder

