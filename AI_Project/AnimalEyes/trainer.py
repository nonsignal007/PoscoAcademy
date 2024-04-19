import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
import time

from torchvision import models , transforms
from torch.utils.data import DataLoader
from data_generator.dataloader import DogEyesDataset
from tempfile import TemporaryDirectory

from tqdm import tqdm
from sklearn.metrics import f1_score

# cfg

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


data_transforms = {
    'train' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ]),
    'val' : transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ]),
}

# image_dataset

train_datasets = DogEyesDataset(transform= data_transforms['train'] , train = True)
val_datasets = DogEyesDataset(transform= data_transforms['val'], train = False)

train_dataloader = DataLoader(train_datasets, batch_size = 4, shuffle = True, num_workers= 4)
val_dataloader = DataLoader(val_datasets, batch_size = 4, shuffle=True , num_workers= 4)

dataloaders = {'train' : train_dataloader,
               'val' : val_dataloader}

dataset_sizes = {'train' : len(train_datasets),
                 'val' : len(val_datasets)}

def train_model(model, criterion, optimizer, scheduler, num_epochs = 10):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        min_loss = 0.0

        for epoch in tqdm(range(num_epochs)):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f}')

                if phase == 'val' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    torch.save(model.state_dict() , best_model_params_path)

            print()
    
        time_elapsed = time.time() - since

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc : {min_loss:4f}')

        model.load_state_dict(torch.load(best_model_params_path))

    return model

if __name__ == '__main__':
        
    # model define
    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 15)

    model_ft = model_ft.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_ft = optim.SGD(model_ft.parameters() , lr = 0.001, momentum = 0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft , step_size= 7 , gamma= 0.1)

    model_ft = train_model(model_ft , criterion=criterion , optimizer= optimizer_ft , scheduler=exp_lr_scheduler , num_epochs= 2)


