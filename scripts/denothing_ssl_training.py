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
from src.augmentation.noise import *


# Training function
def train_epoch_den(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)

    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)

        image_noisy = add_noise(image_batch, noise_factor)

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

        image_noisy = image_noisy.to(device)
        # Encode data
        encoded_data = encoder(image_noisy)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_noisy)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def plot_ae_outputs_den(encoder, decoder, test_dataset, device, n=5, noise_factor=0.3, ):
    plt.figure(figsize=(10, 4.5))
    for i in range(n):

        ax = plt.subplot(3, n, i + 1)
        img = test_dataset[i][0].unsqueeze(0)
        image_noisy = add_noise(img, noise_factor)
        image_noisy = image_noisy.to(device)

        encoder.eval()
        decoder.eval()

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
def test_epoch_den(encoder, decoder, device, dataloader, loss_fn, noise_factor=0.3):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_noisy = add_noise(image_batch, noise_factor)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
        print(f"   >>> Test loss: {val_loss}")
    return val_loss.data


def main():
    #############################
    #       LOAD DATASET        #
    #############################
    data_dir = '../data/dataset'
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    print(f"    >> [SSL CAE Train] Trainset samples: {len(train_dataset)}")
    print(f"    >> [SSL CAE Train] Testset samples: {len(test_dataset)}")

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


    #############################
    #        DATA LOADER        #
    #############################
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


    # Define the geometrical data augmentation transform
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
    combined_train_dataseta = []
    for image, label in train_dataset:
        geometric_image = geometric_transform(image)
        rotation_image = rotation_transform(image)
        combined_train_dataseta.append((image, label))  # Add original image
        combined_train_dataseta.append((geometric_image, label))  # Add blurred image
        combined_train_dataseta.append((rotation_image, label))  # Add blurred image


    combined_train_dataset = []
    i = 0
    for image, label in combined_train_dataseta:
        combined_train_dataset.append((image, label))  # Add original image

        if i % 3 == 0:
            inverted_image = 1 - image
            combined_train_dataset.append((inverted_image, label))  # Add blurred image
        i += 1


    if show_output:
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



    # m = len(train_dataset)
    m = len(combined_train_dataset)

    # random_split randomly split a dataset into non-overlapping new datasets of given lengths
    # train (55,000 images), val split (5,000 images)
    train_data, val_data = random_split(combined_train_dataset, [int(m - m * 0.2), int(m * 0.2)])
    # train_data, val_data = random_split(train_dataset, [int(m - m * 0.1), int(m * 0.1)])

    print(f"    >> [SSL CAE Train] Trainset samples after transform: {len(train_data)}")

    batch_size = 256
    # The dataloaders handle shuffling, batching, etc...
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #############################
    #        MODEL CONFIG       #
    #############################
    # Set the random seed for reproducible results
    torch.manual_seed(0)

    # Initialize the two networks
    d = 4
    encoder = Encoder(encoded_space_dim=d, fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=d, fc2_input_dim=128)


    # Define the loss function
    loss_fn = torch.nn.MSELoss()
    # loss_fn = InfoNCE()

    # Define an optimizer (both for the encoder and the decoder!)
    lr = 0.001  # Learning rate

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'    >>> Selected device: {device}')

    # Choose optimizer
    # optim = torch.optim.Adam(params_to_optimize, lr=lr)
    # optim = torch.optim.RAdam(params_to_optimize, lr=lr)
    optim = torch.optim.NAdam(params_to_optimize, lr=lr, weight_decay=1e-5)
    # optim = torch.optim.RMSprop(params_to_optimize, lr=lr)

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    #############################
    #         TRAINING          #
    #############################
    # Training cycle
    noise_factor = 0.35  # Added noise --> Pretext task SSL
    num_epochs = 50
    history_da = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print('EPOCH %d/%d' % (epoch + 1, num_epochs))
        # Training (use the training function)
        train_loss = train_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optim, noise_factor=noise_factor)

        # Validation  (use the testing function)
        val_loss = test_epoch_den(
            encoder=encoder,
            decoder=decoder,
            device=device,
            dataloader=valid_loader,
            loss_fn=loss_fn, noise_factor=noise_factor)

        # Print Validationloss
        history_da['train_loss'].append(train_loss)
        history_da['val_loss'].append(val_loss)
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        if show_output:
            plot_ae_outputs_den(encoder, decoder, test_dataset, device, noise_factor=noise_factor)


    # Saving the model.
    # torch.save(encoder.state_dict(), "../models/autoencoder_model_1506a.pth")
    torch.save(encoder.encoder_cnn.state_dict(), "../models/autoencoder_model_1506d_enc.pth")
    torch.save(encoder.encoder_lin.state_dict(), "../models/autoencoder_model_1506d_lin.pth")
    plot_ae_outputs_den(encoder, decoder, test_dataset, device, noise_factor=noise_factor)
    test_epoch_den(encoder, decoder, device, test_loader, loss_fn).item()


if __name__ == '__main__':
    main()
