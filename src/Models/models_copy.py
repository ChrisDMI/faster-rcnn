import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.backbone_utils import BackboneWithFPN, resnet_fpn_backbone

from torchvision.models import (
    MobileNet_V2_Weights,
    VGG11_Weights,
    VGG13_Weights,
    VGG16_Weights,
    VGG19_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt101_32X8D_Weights
)



def create_model(num_classes, backbone_name='fasterrcnn_resnet50_fpn'):
    # Define available backbones and their configurations
    backbones = {
    'mobile_net': {'model': torchvision.models.mobilenet_v2, 'out_channels': 1280, 'weights': MobileNet_V2_Weights.DEFAULT},
    'vgg_11': {'model': torchvision.models.vgg11, 'out_channels': 512, 'weights': VGG11_Weights.DEFAULT},
    'vgg_13': {'model': torchvision.models.vgg13, 'out_channels': 512, 'weights': VGG13_Weights.DEFAULT},
    'vgg_16': {'model': torchvision.models.vgg16, 'out_channels': 512, 'weights': VGG16_Weights.DEFAULT},
    'vgg_19': {'model': torchvision.models.vgg19, 'out_channels': 512, 'weights': VGG19_Weights.DEFAULT},
    'resnet_18': {'model': torchvision.models.resnet18, 'out_channels': 512, 'weights': ResNet18_Weights.DEFAULT},
    'resnet_34': {'model': torchvision.models.resnet34, 'out_channels': 512, 'weights': ResNet34_Weights.DEFAULT},
    'resnet_50': {'model': torchvision.models.resnet50, 'out_channels': 2048, 'weights': ResNet50_Weights.DEFAULT},
    'resnet_101': {'model': torchvision.models.resnet101, 'out_channels': 2048, 'weights': ResNet101_Weights.DEFAULT},
    'resnet_152': {'model': torchvision.models.resnet152, 'out_channels': 2048, 'weights': ResNet152_Weights.DEFAULT},
    'resnext101_32x8d': {'model': torchvision.models.resnext101_32x8d, 'out_channels': 2048, 'weights': ResNeXt101_32X8D_Weights.DEFAULT}
}


    # Default to fasterrcnn_resnet50_fpn if backbone is not specified or unsupported
    if backbone_name not in backbones:
        # Load the pre-trained Fast R-CNN model with a ResNet-50 FPN backbone
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(min_size=300, max_size=480, weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

        # Replace the classifier with a new one for custom num_classes (num_classes + background)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        # Custom image normalization parameters for custom backbones
        ft_mean = [0.485, 0.456, 0.406]
        ft_std = [0.229, 0.224, 0.225]

        # Load the specified backbone model
        backbone_model_func = backbones[backbone_name]['model']
        weights = backbones[backbone_name]['weights']
        backbone_model = backbone_model_func(weights=weights)()

        # Configure the backbone (specific to each model)
        if 'resnet' in backbone_name or 'resnext' in backbone_name:
            #returns an iterator over all the layers of the model and 
            # converts this iterator into a list and slices off the last layer, essentially keeping all layers except the last one.
            modules = list(backbone_model.children())[:-1]
            # takes these layers and creates a new sequential module. This sequential module becomes the new backbone that outputs feature 
            ft_backbone = nn.Sequential(*modules)
            # Use resnet_fpn_backbone for ResNet/ResNeXt models
            #backbone = resnet_fpn_backbone(backbone, pretrained=True)
        else:
            #For other models like MobileNet or VGG, the feature extraction layers are typically defined under features.
            ft_backbone = backbone_model.features

        #sets the number of output channels for the backbone. 
        ft_backbone.out_channels = backbones[backbone_name]['out_channels']
        
        # Create the Faster R-CNN model using the specified backbone
        model = FasterRCNN(backbone=ft_backbone, 
                           num_classes=num_classes,
                            image_mean=ft_mean,
                            image_std=ft_std)

    return model

# Example usage of create_model
#num_classes = 5  # Example: 4 classes + 1 background
#model = create_model(num_classes=num_classes, backbone='fasterrcnn_resnet50_fpn')
# If you want to use a custom backbone, replace 'fasterrcnn_resnet50_fpn' with the backbone's name.
