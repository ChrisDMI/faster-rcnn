import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor




def create_model(num_classes):
    
    # Load the pre-trained Fast R-CNN model with a ResNet-50 FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(min_size=300, max_size=480, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the classifier with a new one for custom num_classes (num_classes + background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    return model

# Example usage of create_model
#num_classes = 5  # Example: 4 classes + 1 background
#model = create_model(num_classes=num_classes)

