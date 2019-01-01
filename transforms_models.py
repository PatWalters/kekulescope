from torchvision import transforms
from torchvision import models
import torch.nn as nn

augmented_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.RandomRotation(degrees=90),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

std_transforms = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_model(name):
    if name == "vgg19_bn":
        model_ft = models.vgg19_bn(pretrained=True)
        modules = []
        modules.append(nn.Linear(in_features=25088, out_features=4096, bias=True))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(in_features=4096, out_features=1000, bias=True))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(in_features=1000, out_features=200, bias=True))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(in_features=200, out_features=100, bias=True))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(in_features=100, out_features=1, bias=True))
        classi = nn.Sequential(*modules)
        model_ft.classifier = classi
        return model_ft
