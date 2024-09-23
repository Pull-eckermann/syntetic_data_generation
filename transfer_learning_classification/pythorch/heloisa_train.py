from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from time import localtime, strftime
from json import dump
from sklearn.metrics import roc_curve, auc
from torchvision.transforms import v2

import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
import os

def main():
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    transform_resize = (128, 128)
    LEARNING_RATE = 0.0001
    num_epochs = 15
    batch_size = 32
    frozen_network = False
    normalized = False
    train_dir = '../../../Datasets/PKLot/Heloisa-set/Train'
    val_dir = '../../../Datasets/PKLot/Heloisa-set/Validation'
    test_dir = '../../../Datasets/CNRPark-EXT/Heloisa-set/Test'
    output_dir = 'pklot-results/'
    no_save = False

    if not no_save:
        time = strftime('%d/%m/%Y at %H:%M:%S', localtime())
        with open(f'{output_dir}log.txt', 'a') as file:
            file.write(f'{time} - Configurando modelo\n')

    ## Configurações iniciais
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## Modelo Pré-treinado na imagenet

    ## Transformações para as imagens
    if normalized:
        transform = transforms.Compose([
            transforms.Resize(transform_resize),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(transform_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    ## Carregar datasets
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    ### Definindo a MobileNet
    model = models.mobilenet_v3_large(weights=weights)

    if not frozen_network:
        for param in model.parameters():
            param.requires_grad = True

        #for module in model.modules():
        #    if isinstance(module, nn.BatchNorm2d):
        #        if hasattr(module, 'weight'):
        #            module.weight.requires_grad_(False)
        #        if hasattr(module, 'bias'):
        #            module.bias.requires_grad_(False)
        #        module.eval()
    else:
        for param in model.parameters():
            param.requires_grad = False


    num_classes = len(train_dataset.classes)
    model.classifier[3] = nn.Linear(model.classifier[0].out_features, num_classes)
    #model.classifier = nn.Sequential(
    #    nn.Dropout(p=0.2, inplace=True),
    #    nn.Linear(model.classifier[0].in_features, num_classes),
    #)

    model = model.to(device)

    # Critério e Otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if not no_save:
        time = strftime('%d/%m/%Y at %H:%M:%S', localtime())
        with open(f'{output_dir}log.txt', 'a') as file:
            file.write(f'{time} - Iniciando treinamento\n')

    # Treinamento da rede
    best_validation_loss = 1000
    best_model = None
    best_model_epoch = 1
    model_epoch_data = {}

    empty = 0
    occupied = 0

    for epoch in range(num_epochs):
        total_loss = 0
        model_epoch_data[epoch+1] = {}
        model.train()
        
        total_train_data = 0
        total_val_data = 0
        total_test_data = 0

        for images, labels in train_loader:
            for label in labels:
                if label:
                    occupied += 1
                else:
                    empty += 1
            total_train_data += len(images)
            images, labels = images.to(device), labels.to(device)

            for img in images:
                if random.random() <= 0.1:
                    h, w, _ = img.shape
                    h = int(0.7*h)
                    w = int(0.7*w)
                    transform_img = v2.RandomChoice([
                        v2.RandomHorizontalFlip(p=1.),
                        v2.RandomAutocontrast(p=1.),
                        v2.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5.)),
                        v2.RandomCrop((h,w))
                    ])
                    img = transform_img(img)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        if not no_save:
            time = strftime('%d/%m/%Y at %H:%M:%S', localtime())
            training_loss = total_loss / len(train_loader)
            model_epoch_data[epoch+1]['training_loss'] = training_loss 
            log_msg = f'\n{time} - Epoch {epoch+1} finished training.\n' \
                    f'\tTraining Loss: {training_loss}\n' \
                    f'\n\tEmpty Training Spots: {empty}\n' \
                    f'\tOccupied Training Spots: {occupied}\n' \
                    f'\tTotal Training Data: {total_train_data}\n'
            with open(f'{output_dir}log.txt', 'a') as file:
                file.write(log_msg)

        # Avaliação no conjunto de validação
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_predictions = 0
            total_loss = 0

            empty = 0
            occupied = 0
            first_it = True

            for images, labels in val_loader:
                for label in labels:
                    if label:
                        occupied += 1
                    else:
                        empty += 1
                total_val_data += len(images)

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predictions = torch.max(outputs.data, 1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                if first_it:
                    first_it = False
                    epoch_probabilities = nn.functional.softmax(outputs, dim=1)
                    epoch_labels = labels
                else:
                    new_probabilities = nn.functional.softmax(outputs, dim=1)
                    epoch_probabilities = torch.cat((epoch_probabilities, new_probabilities))
                    epoch_labels = torch.cat((epoch_labels, labels))

            if not no_save:
                time = strftime('%d/%m/%Y at %H:%M:%S', localtime())
                validation_loss = total_loss / len(val_loader)
                validation_accuracy = correct_predictions / total_predictions
                model_epoch_data[epoch+1]['validation_loss'] = validation_loss 
                model_epoch_data[epoch+1]['validation_accuracy'] = validation_accuracy 
                log_msg = f'{time} - Epoch {epoch+1} finished validation.\n' \
                        f'\tValidation Accuracy: {100 * validation_accuracy}%\n' \
                        f'\tValidation Loss: {validation_loss}\n' \
                        f'\n\tEmpty Validation Spots: {empty}\n' \
                        f'\tOccupied Validation Spots: {occupied}\n' \
                        f'\tTotal Validation Data: {total_val_data}\n'

            if (not no_save) and (validation_loss < best_validation_loss):
                best_model = model

                best_validation_loss = validation_loss 
                best_model_epoch = epoch+1
                best_model_optimizer = optimizer.state_dict()
                best_model_criterion = criterion
                best_model_probabilities = epoch_probabilities.clone()
                best_model_labels = epoch_labels.clone()

                try:
                    with open(f'{output_dir}best_model_info.json', 'w') as file:
                        dump({'epoch': epoch+1} | model_epoch_data[epoch+1], file)
                    torch.save(model.state_dict(),
                    #torch.save(model,
                            f'{output_dir}best_model.pt')
                    torch.save(best_model_probabilities,
                            f'{output_dir}best_model_probabilities.pt')
                    torch.save(best_model_labels,
                            f'{output_dir}best_model_labels.pt')
                    torch.save(optimizer.state_dict(),
                            f'{output_dir}best_model_optimizer.pt')
                    torch.save(criterion,
                            f'{output_dir}best_model_criterion.pt')
                except Exception as e:
                    print(e)

            try: 
                if not no_save:
                    with open(f'{output_dir}log.txt', 'a') as file:
                        file.write(log_msg)
                    with open(f'{output_dir}model_epoch_data.json', 'w') as file:
                        dump(model_epoch_data, file)
            except Exception as e:
                print(e)
                print(log_msg)

    if no_save:
        exit()

    print(f'Best Model: epoch {best_model_epoch}')

#   np.save("best_model_probabilities", best_model_probabilities)
    y_score = best_model_probabilities[:, 1].cpu().detach().numpy()
    np.save("y_score", y_score)
    fpr, tpr, thresholds = roc_curve(best_model_labels.cpu(), y_score)
    eer_threshold = thresholds[np.nanargmin(np.abs(tpr - (1 - fpr)))]
    eer = fpr[np.abs(thresholds - eer_threshold).argmin()]

    eer_file = {}
    eer_file["eer_threshold"] = eer_threshold
    try:
        with open(f'{output_dir}eer_threshold.json', 'w') as file:
            dump(eer_file, file)
    except Exception as e:
        print(e)
        print(eer_file)
        eer_file["eer_threshold"] = str(eer_threshold)
        with open(f'{output_dir}eer_threshold.json', 'w') as file:
            dump(eer_file, file)
       

    log_msg = f'\n\nEER: {eer}\n' \
            f'EER threshold: {eer_threshold}'
    try:
        with open(f'{output_dir}log.txt', 'a') as file:
            file.write(log_msg)
    except Exception as e:
        print(e)
        print(log_msg)

    predictions = [1 if i >= eer_threshold else 0 for i in y_score]

    best_model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    best_model.classifier[3] = nn.Linear(best_model.classifier[0].out_features, num_classes)
    best_model.load_state_dict(torch.load(f'{output_dir}best_model.pt'))
    #best_model = torch.load(f'{output_dir}best_model.pt')
    best_model = best_model.to(device)
    best_model.eval()

    #tests = os.listdir(test_dir)
    tests = ['CNRPark-EXT']
    for test in tests:
        #test_files = test_dir + test
        test_files = test_dir

        correct_predictions = 0
        total_predictions = 0
        total_loss = 0

        empty = 0
        occupied = 0
        total_test_data = 0

        test_dataset = ImageFolder(root=test_files, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


        with torch.no_grad():
            for images, labels in test_loader:
                for label in labels:
                    if label:
                        occupied+=1
                    else:
                        empty+=1
                total_test_data += len(images)

                images, labels = images.to(device), labels.to(device)

                outputs = best_model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probabilities = nn.functional.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
                predictions = np.array([1 if i >= eer_threshold else 0 for i in probabilities])
                labels_array = np.array(labels.cpu().detach().numpy())
                correct_predictions += (predictions == labels_array).sum().item()
                total_predictions += labels.size(0)

        testing_loss = total_loss / len(test_loader)
        testing_accuracy = correct_predictions / total_predictions
        time = strftime('%d/%m/%Y at %H:%M:%S', localtime())
        log_msg = f'\n{time} - Finished ' + test + f' testing. \n\tTesting Loss: {testing_loss}' \
                f'\n\tTesting Accuracy: {100 * testing_accuracy}%\n' \
                f'\n\tEmpty Testing Spots: {empty}\n' \
                f'\tOccupied Testing Spots: {occupied}\n' \
                f'\tTotal Testing Data: {total_test_data}' \
                f'\n'
        try:
            with open(f'{output_dir}log.txt', 'a') as file:
                file.write(log_msg)
        except Exception as e:
            print(e)
            print(log_msg)

def tensorToImage(tensor):
    image = tensor.cpu().detach

if __name__ == '__main__':
    main()