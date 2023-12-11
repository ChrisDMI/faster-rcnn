import os
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import glob
import cv2
import torch
from Dataloaders import dl_image
from Models import models
from utils import img_transform, plot_image, inference



def get_image_paths(image_folder):
    # Adjust the pattern as needed to include other image formats
    image_formats = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for image_format in image_formats:
        image_paths.extend(glob.glob(os.path.join(image_folder, image_format)))
    return image_paths






def setup_output_directories():
    # Get the current directory
    current_dir = os.getcwd()

    # Fetch current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

    # Create the predict directory name
    predict_batch_dir_name = "predict-batch-" + dt_string

    # Construct the PREDICT_BATCH_DIR path in the predict folder of the parent directory
    PREDICT_BATCH_DIR = os.path.join(current_dir, 'predict-batch', predict_batch_dir_name)


    # If the specific output training directory doesn't exist, create it
    if not os.path.exists(PREDICT_BATCH_DIR):
        os.makedirs(PREDICT_BATCH_DIR)
    return PREDICT_BATCH_DIR




def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("We use the following device: ", device)
    model = models.create_model(num_classes=dl_image.NUM_CLASSES)
    model.to(device)
    checkpoint_dir = dl_image.PATH_TO_TRAIN_MODEL

    print('\n=========================================')
    print(f'This is the path of your loaded model for prediction : \n {checkpoint_dir}')
    print('=========================================\n')

    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode
    return model, device

def predict_batch(model, device, classes_list, img_folder, PREDICT_BATCH_DIR):
    img_paths_to_predict = get_image_paths(img_folder)
    print("========================= Starting precdiction on each image =========================")
    for img_path in img_paths_to_predict:
        img = cv2.imread(img_path)
        img = img_transform(img)
        img_tensor = img.unsqueeze(0).to(device)  # Add batch dimension and send to device

        # Perform prediction
        with torch.no_grad():
            predictions = model(img_tensor)
            prediction = predictions[0]

        # Extract image name and create a directory for its outputs
        img_name = Path(img_path).stem
        img_output_dir = os.path.join(PREDICT_BATCH_DIR, img_name)
        os.makedirs(img_output_dir, exist_ok=True)

        # Convert tensor to numpy array for plotting and processing
        img_np = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

        boxes = prediction['boxes'].cpu()
        scores = prediction['scores'].cpu()
        labels = prediction['labels'].cpu()
        
        # Save the plot of the image with predictions
        plot_image(img_np, boxes, scores, labels, classes_list, save_path=os.path.join(img_output_dir, "inference.png"))

        # Save cropped images for each prediction
        for i, (box, label) in enumerate(zip(prediction['boxes'], prediction['labels'])):
            label_folder = os.path.join(img_output_dir, classes_list[label.item()])
            os.makedirs(label_folder, exist_ok=True)
            xmin, ymin, xmax, ymax = map(int, box)
            cropped_img = img_np[ymin:ymax, xmin:xmax]
            cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
            cropped_img_path = os.path.join(label_folder, f"cropped_{classes_list[label]}_{i}.png")
            cropped_img.save(cropped_img_path)


    print("Predictions and cropped images saved in the respective folders.")



if __name__ == '__main__':
    PREDICT_BATCH_DIR = setup_output_directories()
    model, device = load_model()

    image_folder = dl_image.BATCH_IMAGE_FOLDER  # This should be the path to the folder containing images
    predict_batch(model, device, dl_image.CLASSES_LIST, image_folder, PREDICT_BATCH_DIR)