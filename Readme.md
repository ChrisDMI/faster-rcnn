# Faster R-CNN Project

Faster R-CNN (Region-based Convolutional Neural Networks) is a powerful, state-of-the-art deep learning algorithm used for object detection tasks. It identifies both the location of objects in images and classifies those objects. This repository focuses on training and utilizing the Faster R-CNN model for custom object detection tasks.

## Setting Up the Development Environment

To train and use the Faster R-CNN model, follow these steps to set up your development environment:

### Prerequisites

- Python 3.8 or above
- Access to a GPU for faster training (optional but recommended)

### Virtual Environment Setup

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv faster_rcnn_env
   ```
2. **Activate the Virtual Environment**:
   - On Windows: `faster_rcnn_env\Scripts\activate`
   - On macOS/Linux: `source faster_rcnn_env/bin/activate`

3. **Install Dependencies**:
   Navigate to the project's root directory and run:
   ```bash
   pip install -r src/requirements.txt
   ```

### PyTorch Installation

Choose one of the following commands based on your CUDA version to install PyTorch:

- **For CUDA 11.8**:
  - Using Conda: `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
  - Using Pip: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`

- **For CUDA 12.1**:
  - Using Conda: `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia`
  - Using Pip: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

### Preparing the Dataset

1. **CSV File**: Create a CSV file with columns: `Frames`, `Xmin`, `Ymin`, `Xmax`, `Ymax`, and `Labels`. This file will contain the annotations for your training data. (See `data/dataset.csv` file)
2. **Labels File**: Create a `labels.txt` file with each class name on a new line, starting with 'background' as the first line.
3. **Data Directory**: Place your image data in the `data/img/` folder and upload the `.csv` and `.txt` files to the `data/` folder.

## Training the Model

1. **Configure Hyperparameters**: Set your desired hyperparameters and directory paths in the `src/Config/config.yaml` file.
2. **Start Training**: From the project's root directory, launch the training process:
   ```bash
   python src/train.py
   ```
   This process will create an `output` folder with subfolders named `train-{day}-{time}` for each training session. Inside these, there are `train` and `test` folders. In `train` folder you'll find the trained model and loss plots and in `test` folder you will find predictions on a certain number of images that you can set in `src/Config/config.yaml` file.

## Making Predictions

- **Single Image Prediction**: Run `python src/predict.py` to predict on an individual image. This will generate a `predict` folder with subfolders named `predict-{day}-{time}` containing the predicted image and cropped class images.
- **Batch Image Prediction**: Run `python src/predict_batch.py` for batch predictions. This creates a `predict-batch` folder with similar subfolders, containing predictions for each image in the specified batch.

Feel free to contribute to this project and enhance its capabilities. For any queries or contributions, please refer to the contact details provided. 

---

*Note: This README assumes a basic understanding of object detection concepts and familiarity with Python programming. For detailed instructions or troubleshooting, refer to the official Faster R-CNN documentation and PyTorch guides.*


## Author

**Christian Segnou**
- Email: nguiepemarius@gmail.com
- LinkedIn: https://www.linkedin.com/in/christian-segnou-ph-d-4461b2102/

Feel free to reach out to me for collaborations or just a friendly chat!
