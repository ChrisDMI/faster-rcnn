import os
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
import cv2
import torch
from Dataloaders import dl_image
from Models import models
from utils import img_transform, plot_image, inference


#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'






def setup_output_directories():
    # Get the current directory
    current_dir = os.getcwd()

    # Fetch current date and time
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

    # Create the predict directory name
    predict_dir_name = "predict-" + dt_string

    # Construct the PREDICT_DIR path in the predict folder of the parent directory
    PREDICT_DIR = os.path.join(current_dir, 'predict', predict_dir_name)

    # If the predict directory doesn't exist, create it
    if not os.path.exists(os.path.join(current_dir, 'predict')):
        os.makedirs(os.path.join(current_dir, 'predict'))

    # If the specific output training directory doesn't exist, create it
    if not os.path.exists(PREDICT_DIR):
        os.makedirs(PREDICT_DIR)

    return PREDICT_DIR


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




def predict(model, device, classes_list, img_path_to_predict, PREDICT_DIR):

    img = cv2.imread(img_path_to_predict)
    img = img_transform(img)
    img = img.to(device)

    # Load last checkpoint
    # CHANGE THE OUTPUT_DIR IF CKPT IS STORED ELSEWHERE

    img_tensor = img.unsqueeze(0).to(device)  # Add batch dimension and send to device

    # Perform prediction
    with torch.no_grad():
        predictions = model(img_tensor)
        prediction = predictions[0]

    print("========================= Starting precdiction on one image =========================")
    #boxes, scores, labels = inference(img, model, device)
    # Convert tensor to numpy array for plotting and processing

    boxes = prediction['boxes'].cpu()
    scores = prediction['scores'].cpu()
    labels = prediction['labels'].cpu()
   

    img_np = img.squeeze(0).cpu().permute(1,2,0).numpy()

    plot_image(img_np, boxes, scores, labels, classes_list, save_path=os.path.join(PREDICT_DIR, f"inference.png"))
    print("Prediction saved in predict folder.")

    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Create folder for each label if it doesn't exist
        label_folder = os.path.join(PREDICT_DIR, classes_list[label])
        os.makedirs(label_folder, exist_ok=True)

        # Cropping and saving the image
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_img = img_np[ymin:ymax, xmin:xmax]
        cropped_img = Image.fromarray((cropped_img * 255).astype(np.uint8))
        cropped_img_path = os.path.join(label_folder, f"cropped_{classes_list[label]}_{i}.png")
        cropped_img.save(cropped_img_path)

    print("Cropped images saved in corresponding label folders.")



if __name__ == '__main__':

    PREDICT_DIR = setup_output_directories()

    model, device = load_model()

    predict(model, device, dl_image.CLASSES_LIST, dl_image.IMAGE_NAME_TO_PREDICT, PREDICT_DIR)