from tqdm import tqdm
import sys

#sys.path.append('c:/users/blanchard/appdata/roaming/python/python39/site-packages')
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
import torch.optim as optim
from segmentation_pytorch import UNET
#import semanticseg.DataDictModule
from utils import (load_checkpoint, save_checkpoint, check_accuracy,get_loaders, save_predictions_as_imags)
#from semanticseg.DataDictModule import DataDictModule

## Hyperparameters 
LEARNIN_RATE =1E-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS= 200
NUM_WORKERS = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True

TRAIN_IMG_DIR  = "E:\\images_train"
TRAIN_MASK_DIR  = "E:\\Labels_train"
VAL_IMG_DIR  = "E:\\images_val"
VAL_MASK_DIR  = "E:\\Labels_val"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop) : 
        data = data.to(device = DEVICE)
        #targets = targets.float().unsqueeze(1).to(device= DEVICE)
        targets = targets.float().unsqueeze(1).to(device= DEVICE)
        #num_classes = 4
        #targets = torch.nn.functional.one_hot(targets.long(), num_classes)
        #targets = targets.permute(0, 3, 1, 2)

        #FORWARD 
        with torch.cuda.amp.autocast() : 
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #BACKWARD
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop 
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0],
        max_pixel_value = 255.0,
        ),
        ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
        
        A.Normalize(
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0],
        max_pixel_value = 255.0,
        ),
        ToTensorV2(),
        ],
    )
    if torch.cuda.device_count() > 1:
        model = UNET(in_channels = 3, out_channels = 1)
        model = nn.DataParallel(model)
        model.to(DEVICE)
    else:
        model = UNET(in_channels = 3, out_channels = 1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNIN_RATE)
    train_loader, val_loader  = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR, 
        VAL_MASK_DIR, 
        BATCH_SIZE, 
        train_transform, 
        val_transform,
        NUM_WORKERS, 
        PIN_MEMORY)
    

    #if LOAD_MODEL : 
        #load_checkpoint("C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\01-promenades des hauteurs\\3mm\\trainmy_checkpoint.pth.tar", model)
    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS) :
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        #save the model 
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        print("##################",epoch,"##############################")
        #check_ accuracy 
        check_accuracy(val_loader,model, device=DEVICE)
        torch.save(model.state_dict(), "E:\\model_5mm_352Im.pth.tar")

        #print some exemple to a folder
        save_predictions_as_imags(val_loader, model,folder ="C:\\Users\\ABDEL\\Desktop\\POURABDEL\\Donnees d'entrainnement\\01-promenades des hauteurs\\3mm\\resultat_predict\\", device= DEVICE )




if __name__ =="__main__" : 
    main()