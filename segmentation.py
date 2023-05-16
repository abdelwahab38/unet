import torch
import os
from dataset import seg_data
from torch.utils.data import DataLoader
from utils import save_segmented_image, get_image_paths
import torch.nn as nn
from segmentation_pytorch import UNET


model = UNET(in_channels=3, out_channels=1)

image_dir ="C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\01-promenades des hauteurs\\rue 1\\images"
model_path = "C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\train_1\\model.pth.tar"
output_dir = "C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\01-promenades des hauteurs\\rue 1\\sortie"


model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()
image_paths = get_image_paths(image_dir)
test_dataset = seg_data(image_paths)
print (image_paths)


for i, image_path in enumerate(image_paths):
    # Charger l'image
    image = test_dataset[i]

    # Effectuer la segmentation sur l'image avec votre modèle
    image = image.unsqueeze(0)
    segmented_image = model(image)

    # Obtenez le nom de l'image d'entrée
    image_name = os.path.basename(image_path)

    # Enregistrer l'image segmentée avec le même nom que l'image d'entrée
    output_path = os.path.join(output_dir, image_name)
    save_segmented_image(segmented_image, output_path)