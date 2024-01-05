import os
from datetime import datetime
import torch
from Dataloaders import dataset
from utils import load_config, classes_list
from torch.utils.data import Dataset

config = load_config(r'./src/Config/config.yaml')

# Get the current directory
current_dir = os.getcwd()
IMG_PATH = os.path.join(current_dir, 'data', config['common']['img_path'])  # Path to the 'img' folder
CSV_PATH = os.path.join(current_dir, 'data', config['common']['csv_file'])  # Path to 'labels.csv' in 'Data'
CLASSES_LIST = classes_list(os.path.join(current_dir, 'data', config['common']['classes_file']))  # Path to 'receipt_labels.txt' in 'Data'
NUM_CLASSES = len(CLASSES_LIST)



PATH_TO_TRAIN_MODEL = config['predict']['path_to_trained_model']
BATCH_IMAGE_FOLDER = config['predict']['batch_image_folder']
IMAGE_NAME_TO_PREDICT = config['predict']['image_to_predict']




SEED = config['common']['seed']

TEST_SIZE = config['train']['test_size']

BATCH_SIZE = config['train']['batch_size']

NMS_THRESH = config['train']['nms_thresh']

#For training

NUM_EPOCHS = config['train']['num_epochs']

LR = config['train']['lr']

LR_MOMENTUM = config['train']['momentum']

LR_DECAY_RATE = config['train']['decay']

LR_SCHED_STEP_SIZE = config['train']['lr_step_size']

LR_SCHED_GAMMA = config['train']['lr_gamma']

NUM_TEST_IMAGES = config['train']['num_test_images']



class DataDetectionLoader(Dataset):

    def __init__(self, model):
        self.model = model



    # construct an optimizer
    def optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=LR, momentum=LR_MOMENTUM, weight_decay=LR_DECAY_RATE)

        # create a learning rate scheduler
        # TODO: step size to be tuned !
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, \
                                                        step_size=LR_SCHED_STEP_SIZE, \
                                                        gamma=LR_SCHED_GAMMA)
        return optimizer, lr_scheduler
    

    # create a train- and validation-dataset with our vehicleDataset
    # split the dataset in train and test set   

    def load_train_val_datasets(self):
        print("================================ Loading training dataset : ===============================")
        data_train = dataset.MyDataset(IMG_PATH, CSV_PATH, CLASSES_LIST, transforms = dataset.Transformation()) 
        print("================================ Loading validation dataset : ===============================")
        data_val = dataset.MyDataset(IMG_PATH, CSV_PATH, CLASSES_LIST, transforms = None) 

        torch.manual_seed(SEED)
        indices = torch.randperm(len(data_train)).tolist()

        test_size = int(len(data_train) * TEST_SIZE)
        dataset_train = torch.utils.data.Subset(data_train, indices[:-test_size])
        dataset_val = torch.utils.data.Subset(data_val, indices[-test_size:])

        
        def collate_fn(batch):
            return tuple(zip(*batch))

        # define training and validation data loaders
        data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

        data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

        return data_loader_train, data_loader_val, dataset_train, dataset_val


