import os
import random
from datetime import datetime
from tqdm import tqdm
import torch
import pickle
from utils import plot_loss, plot_image, inference
from Models import models
from Dataloaders import dl_image

def setup_output_directories():
    # Get the current working directory
    current_dir = os.getcwd()

    # Generate a timestamp string for naming output directories
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    # Create a unique directory name for this training session
    output_dir_name = f"train-{dt_string}"

    # Construct the full path for the output directory
    output_dir = os.path.join(current_dir, 'Output', output_dir_name)

    # Define subdirectories for training and validation results
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    # Create the output, training, and validation directories if they don't exist
    for dir in [output_dir, train_dir, val_dir]:
        os.makedirs(dir, exist_ok=True)

    return train_dir, val_dir

def initialize_model():
    # Determine if CUDA (GPU) is available and select the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the Faster R-CNN model with the specified number of classes
    model = models.create_model(num_classes=dl_image.NUM_CLASSES)

    # Move the model to the selected device (GPU or CPU)
    model.to(device)
    
    return model, device

def load_latest_checkpoint(train_dir, model, optimizer):
    # List all .pth files (model checkpoints) in the training directory
    checkpoint_files = [f for f in os.listdir(train_dir) if f.endswith('.pth')]

    # Initialize the training start epoch and loss dictionary
    start_epoch = 0
    loss_dict = {'train_loss': [], 'valid_loss': []}

    # If there are checkpoints available, load the most recent one
    if checkpoint_files:
        checkpoint_files.sort()
        latest_checkpoint = os.path.join(train_dir, checkpoint_files[-1])
        checkpoint = torch.load(latest_checkpoint)

        # Load the model state, optimizer state, epoch, and loss history from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss_dict = checkpoint['loss_dict']
    
    return start_epoch, loss_dict

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    # Set the model to training mode
    model.train()

    # Initialize an empty list to store loss values for this epoch
    train_loss_list = []

    # Create a tqdm progress bar for this epoch
    tqdm_bar = tqdm(data_loader, total=len(data_loader), desc=f"Epoch {epoch+1}")

    # Iterate over each batch in the data loader
    for idx, (images, targets) in enumerate(tqdm_bar):
        # Move images and targets to the selected device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero out gradients before the forward pass
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing images and targets to the model
        losses = model(images, targets)

        # Compute the total loss for the batch
        loss = sum(losses.values())

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Perform a single optimization step (parameter update)
        optimizer.step()

        # Record the loss value for this batch
        train_loss_list.append(loss.item())

        # Update the progress bar with the current loss value
        tqdm_bar.set_postfix(loss=loss.item())

        # Update the progress bar description with the current loss
        tqdm_bar.set_description(desc=f"Training Loss: {loss:.3f}")

    return train_loss_list




def evaluate(model, data_loader, device):
    """
    Evaluates the model on a given dataset.

    Args:
    - model (torch.nn.Module): The model to be evaluated.
    - data_loader (torch.utils.data.DataLoader): The DataLoader for the dataset to be evaluated.
    - device (torch.device): The device on which to perform the evaluation (e.g., 'cuda' or 'cpu').

    Returns:
    - List[float]: A list of loss values calculated for each batch in the dataset.
    """

    # Initialize an empty list to store loss values
    validation_loss_list = []

    # Iterate over the validation dataset
    tqdm_bar = tqdm(data_loader, total=len(data_loader), desc="Validation")
    for images, targets in tqdm_bar:
        # Move images and targets to the selected device (GPU or CPU)
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        # Disable gradient calculation as it's not needed for validation
        with torch.no_grad():
            # Compute the model's output and calculate losses
            losses = model(images, targets)

        # Sum the losses for the batch and get the scalar value
        total_loss = sum(loss for loss in losses.values())
        loss_value = total_loss.item()

        # Append the scalar loss value for this batch to the list
        validation_loss_list.append(loss_value)

        # Update the progress bar description with the current loss
        tqdm_bar.set_description(f"Validation Loss: {total_loss:.4f}")

    return validation_loss_list




def train(model, device, train_dir, val_dir, data_loader_train, data_loader_val):
    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=dl_image.LR, momentum=dl_image.LR_MOMENTUM, weight_decay=dl_image.LR_DECAY_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=dl_image.LR_SCHED_STEP_SIZE, gamma=dl_image.LR_SCHED_GAMMA)

    # Load the latest checkpoint if available
    start_epoch, loss_dict = load_latest_checkpoint(train_dir, model, optimizer)

    # Training loop
    for epoch in range(start_epoch, dl_image.NUM_EPOCHS):
        print("---------- Epoch {} ----------".format(epoch+1))
        # Train the model for one epoch and get the list of training losses
        train_loss_list = train_one_epoch(model, optimizer, data_loader_train, device, epoch)

        # Update learning rate scheduler after each epoch
        lr_scheduler.step()

        # Evaluate the model on the validation set and get the list of validation losses
        val_loss_list = evaluate(model, data_loader_val, device)

        # Append the current epoch's training and validation losses to the history
        loss_dict['train_loss'].extend(train_loss_list)
        loss_dict['valid_loss'].extend(val_loss_list)

        # Save the model checkpoint after each epoch
        ckpt_file_name = os.path.join(train_dir, f"epoch_{epoch+1}_model.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_dict': loss_dict
        }, ckpt_file_name)

        # Plot and save the loss graphs
        plot_loss(loss_dict['train_loss'], loss_dict['valid_loss'], train_dir)

    # Save the loss history after training is completed
    with open(os.path.join(train_dir, "loss_dict.pkl"), "wb") as file:
        pickle.dump(loss_dict, file)

    print("Training Finished!")

def val(model, device, val_dir, dataset_val, classes_list, num_images):
    # Set the random seed for reproducibility
    random.seed(dl_image.SEED)

    # Loop to generate predictions for a specified number of images from the validation set
    for _ in range(num_images):
        # Randomly select an image from the validation set
        idx = random.randint(0, len(dataset_val) - 1)
        img, target = dataset_val[idx]
        img = img.to(device)

        # Load the last checkpoint for the model
        checkpoint_path = os.path.join(train_dir, f"epoch_{dl_image.NUM_EPOCHS}_model.pth")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Generate predictions for the selected image
        boxes, scores, labels = inference(img, model, device)
        img_np = img.cpu().numpy().transpose(1,2,0)

        # Plot and save the image with predictions
        plot_image(img_np, boxes, scores, labels, classes_list, 
                   save_path=os.path.join(val_dir, f"inference_{idx}.png"))

if __name__ == '__main__':
    train_dir, val_dir = setup_output_directories()
    model, device = initialize_model()
    data_loader_train, data_loader_val, dataset_train, dataset_val = dl_image.DataDetectionLoader(model).load_train_val_datasets()
    
    print('\n===========================================================================')
    print(f"Number of training samples: {int(dl_image.BATCH_SIZE*len(data_loader_train))} | Number of training samples in each batch : {len(data_loader_train)}") 
    print(f"Number of validation samples: {len(data_loader_val)}\n")
    print('===========================================================================')


    print("==================== training data starting ... ====================")
    train(model, device, train_dir, val_dir, data_loader_train, data_loader_val)
    
    val(model, device, val_dir, dataset_val, dl_image.CLASSES_LIST, dl_image.NUM_TEST_IMAGES)