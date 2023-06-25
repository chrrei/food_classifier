import random
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import Food101
from captum.attr import IntegratedGradients

from models.food_resnet18 import FoodResnet18
from models.food_squeezenet import FoodSqueezenet


def classify_image(model, image_path):
    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction, image_path


def classify_random_image(model, test_images_folder: Path):
    image_paths = list(test_images_folder.glob("*"))
    random_image_path = random.choice(image_paths)
    prediction, img = classify_image(model, random_image_path)
    return prediction, img, random_image_path


def classify_all_images(model, test_images_folder: Path):
    image_paths = list(test_images_folder.glob("*.jpg"))
    predictions = []

    for image_path in image_paths:
        prediction, img = classify_image(model, image_path)
        predictions.append(prediction)

    return predictions, image_paths


def plot_images(images, predictions):
    classes_dict = get_class_name_dict()
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        img = Image.open(images[i])
        img.thumbnail((200, 200))
        ax.imshow(img)
        ax.set_title(f"Pred: {classes_dict[predictions[i]]}")
        ax.axis("off")
    plt.show()


def plot_class_activation(model, images, predictions):
    # Initialize the IntegratedGradients method
    ig = IntegratedGradients(model)
    classes_dict = get_class_name_dict()

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        img = Image.open(images[i])
        img.thumbnail((200, 200))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        attr = ig.attribute(
            img_tensor,
            target=predictions[i]
        ).squeeze().detach().numpy().transpose(1, 2, 0)

        # Normalize the activation map
        attr = (attr - attr.min()) / (attr.max() - attr.min())

        # Overlay the activation map on the original image (scaled to [0, 1])
        # 8% original, 92% activation map
        overlay = np.uint8(
            255 * (0.08 * img_tensor.squeeze().permute(1, 2, 0).numpy()
                   + 0.92 * attr)
        )

        ax.imshow(overlay)
        ax.set_title(f"class: {classes_dict[predictions[i]]}")
        ax.axis("off")
    plt.show()


def get_available_models(models_folder: Path):
    model_paths = list(models_folder.glob("*.pt")) + \
                list(models_folder.glob("*.pth")) + \
                list(models_folder.glob("*.onnx"))
    return model_paths


def get_class_name_dict():
    # Load the Food-101 dataset
    food101_dataset = Food101(root='data/', download=True)
    class_to_idx = food101_dataset.class_to_idx
    # Invert the dictionary to map indices to class names
    return {idx: class_name for class_name, idx in class_to_idx.items()}


def exp_main():
    models_folder = Path("models/saved")
    available_models = get_available_models(models_folder)
    print("Available models:")
    for i, file in enumerate(available_models):
        print(f"{i}: {file}")

    chosen_index = int(input("Enter index of the file to load: "))
    model_path = Path(available_models[chosen_index])
    test_images_folder = Path("./experiments/test_images")

    # Load the pre-trained model
    model_string = re.search(r'[/\\]([^/\\_]+)_', str(model_path)).group(1)
    if model_string == 'squeezenet':
        model = FoodSqueezenet(False)
    elif model_string == 'resnet18':
        model = FoodResnet18(False)
    else:
        raise ValueError("Base model was not found.")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Classify all images
    predictions, images = classify_all_images(model, test_images_folder)

    # Plot the images with their predicted classes
    plot_images(images, predictions)

    # Plot the class activation using Captum
    plot_class_activation(model, images, predictions)
