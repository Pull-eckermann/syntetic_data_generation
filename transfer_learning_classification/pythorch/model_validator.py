import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
base_path = "saved_models/"
thresh_path = "eer_threshods/"

model_name = 'PKLot'
#model_name = 'CNR'
#model_name = 'PKLot-mixed'
#model_name = 'CNR-mixed'

print("###############################################################################")
print("Model {} is being loaded for tests".format(model_name))
# Restore the model
import_path = base_path + model_name + '.pth'
thresh_path = thresh_path + model_name + '.txt'

# Load the model
model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[0].out_features, 2)
model.load_state_dict(torch.load(import_path))

#model = torch.load(import_path)

with open(thresh_path, 'r') as thresh_file:
    threshold = float(thresh_file.read())

validation_dir = sys.argv[1]
data_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_dataset = datasets.ImageFolder(validation_dir, data_transforms)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Predict
print(">>>>>> Executing evaluation")
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities = nn.functional.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        preds = np.array([1 if i >= threshold else 0 for i in probabilities])
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)

all_labels = np.array(all_labels).ravel()
all_preds = np.array(all_preds).ravel()

print("Accuracy: ", accuracy_score(all_labels, all_preds))
print("====================================================================================")
