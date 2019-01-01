#!/usr/bin/env python

import sys

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from load_images import *
import multiprocessing

from train_model import train_model

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from tqdm import tqdm
import pandas as pd
import scipy
from transforms_models import *

from stratified_split import stratified_split


def generate_images(sdf_name):
    suppl = Chem.SDMolSupplier(sdf_name)
    img_list = []
    pIC50_list = []
    for mol in tqdm(suppl):
        if mol:
            AllChem.Compute2DCoords(mol)
            png_name = mol.GetProp("ChEMBL_ID") + ".png"
            pIC50 = float(mol.GetProp("pIC50"))
            Draw.MolToFile(mol, png_name)
            img_list.append(png_name)
            pIC50_list.append(pIC50)
    return img_list, pIC50_list


def build_model(input_sdf):
    print(f"Running {input_sdf}")
    EPOCHS = 100
    CV_CYCLES = 10
    output_csv = input_sdf.replace(".sdf", "_ks.csv")
    img_list, pIC50_list = generate_images(input_sdf)

    workers = multiprocessing.cpu_count()

    num_cycles = CV_CYCLES
    out_list = []
    for cycle in range(0, num_cycles):
        train_x, val_x, test_x, train_y, val_y, test_y = stratified_split(img_list, pIC50_list)

        train_list = list(zip(train_x, train_y))
        test_list = list(zip(test_x, test_y))
        val_list = list(zip(val_x, val_y))

        print([len(x) for x in (train_x, test_x, val_x)])
        batch_size = 16

        trainloader = torch.utils.data.DataLoader(
            ImageFilelist(paths_labels=train_list, transform=std_transforms),
            batch_size=batch_size, shuffle=False,
            num_workers=workers)

        valloader = torch.utils.data.DataLoader(
            ImageFilelist(paths_labels=val_list, transform=std_transforms),
            batch_size=batch_size, shuffle=False,
            num_workers=workers)

        testloader = torch.utils.data.DataLoader(
            ImageFilelist(paths_labels=test_list, transform=std_transforms),
            batch_size=batch_size, shuffle=False,
            num_workers=workers)

        device = torch.device("cuda")
        dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader}

        model_ft = get_model("vgg19_bn")
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01)
        model_ft = model_ft.to(device)

        criterion = torch.nn.MSELoss()
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.6)

        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=EPOCHS)

        pred = []
        obs = []
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_ft(inputs)
            for i in range(len(labels)):
                pred.append(outputs.data[i])
                obs.append(labels.data[i])

        current_list = []
        for o, p in zip(obs, pred):
            current_list.append([cycle, o.item(), p.item()])
        print(cycle, scipy.stats.pearsonr([x[1] for x in current_list], [x[2] for x in current_list]))
        out_list += current_list

    df = pd.DataFrame(out_list, columns=["cycle", "obs", "pred"])
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    for file_name in sys.argv[1:]:
        build_model(file_name)
