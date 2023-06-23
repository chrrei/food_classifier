import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
from captum.attr import IntegratedGradients


def classify_image(model, image_path, augment=False):
    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        *([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0.1)
        ] if augment else []),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        output = model(img_tensor)
        prediction = torch.argmax(output, dim=1).item()

    return prediction, img


def classify_random_image(model, test_pictures_folder: Path, augment=False):
    image_paths = list(test_pictures_folder.glob("*"))
    random_image_path = random.choice(image_paths)
    prediction, img = classify_image(model, random_image_path, augment)
    return prediction, img, random_image_path


def classify_all_images(model, test_pictures_folder: Path, augment=False):
    image_paths = list(test_pictures_folder.glob("*"))
    predictions = []
    images = []

    for image_path in image_paths:
        prediction, img = classify_image(model, image_path, augment)
        predictions.append(prediction)
        images.append(img)

    return predictions, images


def plot_images(images, predictions):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_title(f"Predicted class: {predictions[i]}")
        ax.axis("off")
    plt.show()


def plot_class_activation(model, images, predictions):
    # Initialize the IntegratedGradients method
    ig = IntegratedGradients(model)

    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        img_tensor = transforms.ToTensor()(images[i]).unsqueeze(0)
        attr = ig.attribute(img_tensor, target=predictions[i]).squeeze().detach().numpy().transpose(1, 2, 0)
        ax.imshow(attr)
        ax.set_title(f"Class activation for class: {predictions[i]}")
        ax.axis("off")
    plt.show()


def get_available_models(models_folder: Path):
    model_paths = list(models_folder.glob("*.pt")) + \
                list(models_folder.glob("*.pth")) + \
                list(models_folder.glob("*.onnx"))
    return model_paths


if __name__ == "__main__":
    models_folder = Path("../models/saved")
    available_models = get_available_models(models_folder)
    print("Available models:")
    for model_path in available_models:
        print(f"- {model_path}")

    chosen_model_path = input("Enter the path to the model you want to use: ")
    model_path = Path(chosen_model_path)
    test_pictures_folder = Path("./experiments/test_pictures")

    # Load the pre-trained model
    model = torch.load(model_path)
    model.eval()

    # Classify all images
    predictions, images = classify_all_images(model,
                                              test_pictures_folder,
                                              augment=True)

    # Plot the images with their predicted classes
    plot_images(images, predictions)

    # Plot the class activation using Captum
    plot_class_activation(model, images, predictions)
