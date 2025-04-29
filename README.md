# Cleanify-ddpm

## Denoising Diffusion Probabilistic Models (DDPM)

This is my implementation of **Denoising Diffusion Probabilistic Models (DDPMs)**, a generative model that creates realistic images by progressively denoising random noise through a learned reverse process. This implementation is based on the paper **"Denoising Diffusion Probabilistic Models"** by **Jonathan Ho**, **Ajay Jain**, and **Pieter Abbeel**.

The link to the paper can be found here: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

## Overview

DDPMs offer an alternative approach to generative models, such as GANs, by using a **probabilistic diffusion process**. The model consists of two main steps:
1. **Forward Diffusion Process**: Start with a clean image and gradually add noise over iterations, turning the data into pure noise.
2. **Reverse Denoising Process**: Train a neural network to reverse this process, progressively recovering the original image from noisy data.

This project aims to implement DDPMs from scratch using **PyTorch**.

## Features

- **Forward Diffusion Process**: Adds Gaussian noise to images iteratively. We move the images from their original data subspace into a simple noise-filled space. The goal is to make the images unrecognizable, and the complex data distribution is transformed into a simple one. This will result with our image being pure random noise, outside the original data subspace.
- **Reverse Denoising Process**: Uses a U-Net-based neural network to predict and remove noise from the image.
- **Training and Sampling**: Train the model using the CIFAR-10 dataset and generate realistic samples.
  
![Demo gif](https://learnopencv.com/wp-content/uploads/2023/01/diffusion-models-unconditional_image_generation-1.gif)

## Dataset

For this project, I will be using the **CIFAR-10** dataset to train the DDPM.

### Rationale

- **Small and manageable**: CIFAR-10 consists of 60,000 32x32 color images across 10 classes, making it lightweight and ideal for training.
- **Widely used benchmark**: Many generative models, including diffusion models and GANs, use CIFAR-10 for comparison allowing easier evaluation of results.
- **Simple yet diverse**: The dataset includes a variety of objects, providing enough diversity for a generative model to learn meaningful features without being overly complex.

Using CIFAR-10 makes it easier to focus on getting the core DDPM implementation correct before scaling up to more complex datasets (CelebA,ImageNet).
