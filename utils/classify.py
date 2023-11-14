import numpy as np
import os.path as osp
import pandas as pd
import argparse

import torch
import torchvision
from torchvision.models.resnet import model_urls


def get_classification_model(device):
    # Load classification model (resnet50 is the only supported one currently)
    model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
    classification_model = torchvision.models.resnet50(pretrained=True).to(device)
    classification_model.eval()
    return classification_model


def classify_batch(x, y, classification_model, df):
    preds = classification_model(x)
    y_hat = torch.argmax(preds, dim=1)

    # Save stats
    sm_yh = preds.max(dim=1).values.cpu().detach().numpy() # Get softmax of predicted class (\ie. highest softmax)
    df_tmp = pd.DataFrame({'y': y.numpy(), 'y_hat': y_hat.cpu().numpy(), 'softmax[y_hat]': sm_yh}, \
    columns=['y', 'y_hat', 'softmax[y_hat]'])
    df = pd.concat([df, df_tmp], ignore_index=True)

    return df