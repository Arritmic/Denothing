import os
import sys
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

import cv2
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from info_nce import InfoNCE
from torch import nn
import torch.nn.functional as F
import torch.optim as optim



sys.path.insert(0, os.path.abspath('..'))
from src.aes.autoencoder import Encoder, Decoder
from src.aes.encoder_classifier import Classifier
from src.augmentation.noise import *


# Training function
def train_epoch_den(model, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    model.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)

    for image_batch, labels in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        # print(image_batch)
        # print(image_batch.shape)
        # input("stop")

        # transform_temp = transforms.ToPILImage()
        # fig, axs = plt.subplots(5, 5, figsize=(8, 8))
        # for ax in axs.flatten():
        #     # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
        #     img = random.choice(image_noisy)
        #     img = transform_temp(img)
        #     ax.imshow(np.array(img), cmap='gist_gray')
        #     # ax.set_title('Label: %d' % label)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.tight_layout()
        # plt.show()

        # print(image_noisy.shape)
        # input("stop")

        # Encode data
        encoded_data = model(image_batch)
        # Decode data

        # Evaluate loss
        loss = loss_fn(encoded_data, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def plot_ae_outputs_den(model, test_dataset, device, n=5, noise_factor=0.3, ):
    plt.figure(figsize=(10, 4.5))
    for i in range(n):

        ax = plt.subplot(3, n, i + 1)
        img = test_dataset[i][0].unsqueeze(0)
        image_noisy = add_noise(img, noise_factor)
        image_noisy = image_noisy.to(device)

        model.eval()

        with torch.no_grad():
            rec_img = decoder(encoder(image_noisy))

        plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(image_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Corrupted images')

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.7,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.3)
    plt.show()


# Testing function
def test_epoch_den(model, device, dataloader, loss_fn, noise_factor=0.3, input_text="Test"):
    # Set evaluation mode for encoder and decoder
    model.eval()

    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, labels in dataloader:
            # Encode data
            encoded_data = model(image_batch)

            # Append the network output and the original image to the lists
            conc_out.append(encoded_data.cpu())
            conc_label.append(labels.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
        print(f"   >>> {input_text} loss: {val_loss}")
    return val_loss.data


def main():

    # load data in
    data_dir = '../data/dataset'
    train_dataset = torchvision.datasets.EMNIST(root=data_dir, split="balanced",
                                                train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_dataset = torchvision.datasets.EMNIST(root=data_dir, split="balanced",
                                               train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    entire_trainset = torch.utils.data.DataLoader(train_dataset, shuffle=True)

    split_train_size = int(0.9 * (len(entire_trainset)))  # use 80% as train set
    split_valid_size = len(entire_trainset) - split_train_size  # use 20% as validation set

    train_dataset, val_set = torch.utils.data.random_split(train_dataset, [split_train_size, split_valid_size])

    print(f'train set size: {split_train_size}, validation set size: {split_valid_size}')

    print(f"    >> [SSL OCRClass Train] Trainset samples: {split_train_size}")
    print(f"    >> [SSL OCRClass Train] Validation samples: {split_valid_size}")
    print(f"    >> [SSL OCRClass Train] Testset samples: {len(test_dataset)}")

    #############################
    #       VISUALIZATION       #
    #############################
    show_output = False
    if show_output:
        cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Video Stream 2", cv2.WINDOW_NORMAL)

    if show_output:
        fig, axs = plt.subplots(5, 5, figsize=(8, 8))
        for ax in axs.flatten():
            # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
            img, label = random.choice(train_dataset)
            ax.imshow(np.array(img), cmap='gist_gray')
            ax.set_title('Label: %d' % label)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Set the train transform
    train_dataset.transform = train_transform
    # Set the test transform
    test_dataset.transform = test_transform

    # Define the blur data augmentation transform
    blur_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.GaussianBlur(kernel_size=3),  # Adjust the kernel size as needed
        transforms.RandomVerticalFlip(p=0.2),
        transforms.GaussianBlur(kernel_size=3),  # Adjust the kernel size as needed
        transforms.ToTensor()
    ])

    geometric_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor()
    ])

    rotation_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(35),
        transforms.ToTensor()
    ])

    # Combine original and augmented images in the training dataset
    combined_train_dataset = []
    for image, label in train_dataset:
        # blurred_image = blur_transform(image)
        geometric_image = geometric_transform(image)
        rotation_image = rotation_transform(image)
        combined_train_dataset.append((image, label))  # Add original image
        # combined_train_dataseta.append((blurred_image, label))  # Add blurred image
        combined_train_dataset.append((geometric_image, label))  # Add blurred image
        combined_train_dataset.append((rotation_image, label))  # Add blurred image

    transform_temp = transforms.ToPILImage()
    fig, axs = plt.subplots(5, 5, figsize=(8, 8))
    for ax in axs.flatten():
        # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
        img, label = random.choice(combined_train_dataset)
        img = transform_temp(img)
        ax.imshow(np.array(img), cmap='gist_gray')
        ax.set_title('Label: %d' % label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    # Create a DataLoader for the combined training dataset
    batch_size = 16  # Adjust the batch size as needed
    combined_train_loader = torch.utils.data.DataLoader(
        combined_train_dataset, batch_size=batch_size, shuffle=True
    )

    # m = len(train_dataset)
    m = len(combined_train_dataset)

    train_data = combined_train_dataset

    print(f"    >> [OCRClassifier Train] Trainset samples after transform: {len(train_data)}")

    batch_size = 256

    # The dataloaders handle shuffling, batching, etc...
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Initialize the networks
    ocr_classifier = Classifier(fc2_input_dim=128)
    # Load the weights into the dictionary


    # Assuming `classifier` is your classifier model

    # Load the weights into the classifier
    ocr_classifier.encoder_cnn.load_state_dict(torch.load("../models/autoencoder_model_1506d_enc.pth"))
    # ocr_classifier.encoder_lin.load_state_dict(torch.load("../models/autoencoder_model_1506b_lin.pth"))

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = InfoNCE()

    # Define an optimizer (both for the encoder and the decoder!)
    lr = 0.0001  # Learning rate

    params_to_optimize = [
        {'params': ocr_classifier.parameters()}
    ]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # optim = torch.optim.Adam(params_to_optimize, lr=lr)
    optim = torch.optim.RAdam(params_to_optimize, lr=lr)
    # optim = torch.optim.NAdam(params_to_optimize, lr=lr, weight_decay=1e-5)
    # optim = torch.optim.RMSprop(params_to_optimize, lr=lr)

    # Move both the encoder and the decoder to the selected device
    ocr_classifier.to(device)

    # Training cycle
    num_epochs = 50
    history_da = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        # Training (use the training function)
        train_loss = train_epoch_den(
            model=ocr_classifier,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim)
        # Validation  (use the testing function)
        val_loss = test_epoch_den(
            model=ocr_classifier,
            device=device,
            dataloader=valid_loader,
            loss_fn=loss_fn,
            input_text="Valid")
        # Print Validationloss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        # plot_ae_outputs_den(encoder, decoder, test_dataset, device, noise_factor=noise_factor)

    # torch.save(encoder.state_dict(), "../models/autoencoder_model_1506a.pth")
    # plot_ae_outputs_den(ocr_classifier, test_dataset, device)
    test_epoch_den(ocr_classifier, device, test_loader, loss_fn).item()


if __name__ == '__main__':
    main()
