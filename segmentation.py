import torch
import os
from dataset import seg_data
from torch.utils.data import DataLoader
from utils import save_segmented_image
import torch.nn as nn
from segmentation_pytorch import UNET

class VotreModele(nn.Module):
    def __init__(self):
        super(VotreModele, self).__init__()
        # Définissez ici l'architecture de votre modèle, en ajoutant les couches nécessaires

    def forward(self, x):
        # Définissez ici la logique de propagation avant (forward pass) de votre modèle
        return x

# Créez une instance de votre modèle
model = UNET()

image_dir ="C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\01-promenades des hauteurs\\rue 1\\images"
model_path = "C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\train_1\\model.pth.tar"
output_dir = "C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\01-promenades des hauteurs\\rue 1\\sortie"

model_weights = torch.load(model_path)
model.load_state_dict(model_weights)
model.eval()
test_dataset = seg_data(image_dir=image_dir, transform=None)
test_dataloader = DataLoader(test_dataset, batch_size=1)


for image in test_dataloader:
    # Effectuez la segmentation sur l'image avec votre modèle
    with torch.no_grad():
        segmented_image = model(image)
    
    # Obtenez le nom de l'image d'entrée
    input_image_name = test_dataset.images[test_dataset.current_index]
    
    # Construisez le chemin de sortie pour l'image segmentée
    output_image_path = os.path.join(output_dir, input_image_name)
    
    # Enregistrez l'image segmentée
    save_segmented_image(segmented_image, output_image_path)