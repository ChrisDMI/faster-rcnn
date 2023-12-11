
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import glob as glob
import yaml




def classes_list(path_to_txt_file):
    with open(path_to_txt_file, "r") as label:
        LABELS = label.readlines()


    for i, _ in enumerate(LABELS):
        LABELS[i] = LABELS[i].replace("\n", "")

    print("This is your label names : ", LABELS)
    return LABELS



def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)



def img_transform(img):
  """
  Transformation of a sample input for infernce.
  """
  #converts the color space of the image from BGR (Blue, Green, Red) to RGB (Red, Green, Blue).
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
  img /= 255.0
  #converts the image, which is currently a NumPy array, into a PyTorch tensor.
  img = torch.from_numpy(img).permute(2,0,1)
  return img




def inference(img, model, device, detection_threshold=0.70):
  '''
  Infernece of a single input image

  inputs:
    img: input-image as torch.tensor (shape: [C, H, W])
    model: model for infernce (torch.nn.Module)
    detection_threshold: Confidence-threshold for NMS (default=0.7)

  returns:
    boxes: bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    labels: class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
    scores: confidence-score (Format [N] => N times confidence-score between 0 and 1)
  '''

  #sets the model to evaluation mode. This is essential for inference as it deactivates layers like dropout and batch normalization that are only active during training.
  model.eval()

  #moves the image tensor to the device (like GPU or CPU) where the model is located.
  img = img.to(device)

  #passes the image through the model to get the output. The image is wrapped in a list because some models expect a batch of images.
  outputs = model([img])

  #The outputs from the model include bounding boxes, confidence scores, and labels for each detected object.
  #.data.cpu().numpy() is used to move the tensors from the GPU (if there was setting up) to the CPU  and convert them to NumPy arrays for easy processing.
  boxes = outputs[0]['boxes'].data.cpu().numpy()
  scores = outputs[0]['scores'].data.cpu().numpy()
  labels = outputs[0]['labels'].data.cpu().numpy()

  boxes = boxes[scores >= detection_threshold].astype(np.int32)
  labels = labels[scores >= detection_threshold]
  scores = scores[scores >= detection_threshold]

  return boxes, scores, labels





def plot_image(img, boxes, scores, labels, dataset, plot = True, save_path=None):
  '''
  Function that draws the BBoxes, scores, and labels on the image.

  inputs:
    img: input-image as numpy.array (shape: [H, W, C])
    boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
    scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
    labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
    dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
  '''

  #sets up a color map for plotting.
  cmap = plt.get_cmap("tab20b")

  #creating a list of colors from the colormap, one for each class label.
  class_labels = np.array(dataset)
  colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]

  # Create figure and axes
  height, width, _ = img.shape
  fig, ax = plt.subplots(1, figsize=(16, 8))
  # Display the image
  ax.imshow(img)


  for i, box in enumerate(boxes):
    class_pred = labels[i]
    conf = scores[i]
    width = box[2] - box[0]
    height = box[3] - box[1]
    rect = patches.Rectangle(
        (box[0], box[1]),
        width,
        height,
        linewidth=2,
        edgecolor=colors[int(class_pred)],
        facecolor="none",
    )
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.text(
        box[0], box[1],
        s=class_labels[int(class_pred)] + " " + str(int(100*conf)) + "%",
        color="white",
        verticalalignment="top",
        bbox={"color": colors[int(class_pred)], "pad": 0},
    )

  # Used to save inference phase results
  if save_path is not None:
    plt.savefig(save_path)

  #plt.show()




def plot_loss(train_loss, valid_loss, TRAIN_DIR):
    '''
    Function to plot training and valdiation losses and save them in `output_dir'
    '''
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    train_ax.plot(train_loss, color='blue')
    train_ax.set_xlabel('Iteration')
    train_ax.set_ylabel('Training Loss')

    valid_ax.plot(valid_loss, color='red')
    valid_ax.set_xlabel('Iteration')
    valid_ax.set_ylabel('Validation loss')

    figure_1.savefig(f"{TRAIN_DIR}/train_loss.png")
    figure_2.savefig(f"{TRAIN_DIR}/valid_loss.png")
