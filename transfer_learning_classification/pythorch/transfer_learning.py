import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchsummary import summary
from tqdm import tqdm
from sklearn.metrics import roc_curve
import numpy as np

#================================================ Vars declaration ================================================

BATCH_SIZE = 32
N_EPOCHS = 15
IMG_SIZE = (128, 128)
LEARNING_RATE = 0.0001
TRAIN_DIR = '../../../Datasets/PKLot/Heloisa-set/Train'
VALIDATION_DIR = '../../../Datasets/PKLot/Heloisa-set/Validation'
EXPORT_PATH = 'saved_models/PKLot.pth'
THRESH_PATH = 'eer_threshods/PKLot.txt'
TRAINABLE = True

#================================================ Input treatment ================================================

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(contrast=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(TRAIN_DIR, data_transforms['train'])
validation_dataset = datasets.ImageFolder(VALIDATION_DIR, data_transforms['val'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(train_dataset.classes)

#================================================ Base model specification ================================================

# Load pre-trained MobileNetV3 model
base_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)

if TRAINABLE:
    for param in base_model.parameters():
        param.requires_grad = True

    #for module in model.modules():
    #    if isinstance(module, nn.BatchNorm2d):
    #        if hasattr(module, 'weight'):
    #            module.weight.requires_grad_(False)
    #        if hasattr(module, 'bias'):
    #            module.bias.requires_grad_(False)
    #        module.eval()
else:
    for param in base_model.parameters():
        param.requires_grad = False

# Replace the classifier with a new one
base_model.classifier[3] = nn.Linear(base_model.classifier[0].out_features, num_classes)
#base_model.classifier = nn.Sequential(
#        nn.Dropout(p=0.2, inplace=True),
#        nn.Linear(base_model.classifier[0].in_features, 1),
#    )

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = base_model.to(device)
#summary(model, (3,) + IMG_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

#================================================ Start training ================================================

best_model_wts = model.state_dict()
best_loss = float('inf')

for epoch in range(N_EPOCHS):
    print(f'Epoch {epoch}/{N_EPOCHS - 1}')
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            data_loader = train_loader
        else:
            model.eval()   # Set model to evaluate mode
            data_loader = validation_loader

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        loop = tqdm(data_loader)
        for idx, (inputs, labels) in enumerate(loop):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # add information to progress bar
            if phase == 'train':
                loop.set_description(f"Training Progress")
            else:
                loop.set_description(f"Validation Progress")

        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Deep copy the model
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

    print()

# Load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), EXPORT_PATH)
#torch.save(model, EXPORT_PATH)

#================================================ Evaluate and compute threshold ================================================

model.eval()
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        probabilities = nn.functional.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities)

all_labels = np.array(all_labels).ravel()
all_probs = np.array(all_probs).ravel()

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]

print("Equal Error Rate threshold: ", eer_threshold)

with open(THRESH_PATH, 'w') as file_out:
    file_out.write(str(eer_threshold))