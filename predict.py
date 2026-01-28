#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py — Predict flower classes from an image and a saved checkpoint.

Features
- Reads an image and a checkpoint to print most likely class and probability
- Optionally prints Top-K classes and probabilities
- Supports mapping class labels to names via a JSON file
- GPU inference via --gpu

Example:
python predict.py image.jpg checkpoints/resnet18_checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
"""

import argparse
import json
import os
from typing import Tuple, List

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


def build_model_from_checkpoint(ckpt: dict) -> nn.Module:
    arch = ckpt['arch']
    info = ckpt['classifier_info']
    num_classes = info['num_classes']
    hidden_units = info['hidden_units']

    if arch == 'resnet18':
        model = models.resnet18(weights=None)
        in_features = info['in_features']
        layers = []
        last = in_features
        for hu in hidden_units:
            layers.append(nn.Linear(last, hu))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            last = hu
        layers.append(nn.Linear(last, num_classes))
        layers.append(nn.LogSoftmax(dim=1))
        model.fc = nn.Sequential(*layers)
    elif arch == 'vgg16':
        model = models.vgg16(weights=None)
        in_features = info['in_features']
        layers = []
        last = in_features
        for hu in hidden_units:
            layers.append(nn.Linear(last, hu))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            last = hu
        layers.append(nn.Linear(last, num_classes))
        layers.append(nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(*layers)
    else:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")

    model.load_state_dict(ckpt['state_dict'])
    model.class_to_idx = ckpt['class_to_idx']
    return model


def process_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm(img)


def predict(image_path: str, checkpoint: str, top_k: int = 5, category_names: str = None, gpu: bool = False) -> Tuple[List[float], List[str]]:
    device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')
    ckpt = torch.load(checkpoint, map_location=device)
    model = build_model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    # Prepare image
    img = process_image(image_path).unsqueeze(0).to(device)

    with torch.no_grad():
        logps = model(img)  # model ends with LogSoftmax
        probs = torch.exp(logps)
        top_prob, top_idx = probs.topk(top_k, dim=1)

    top_prob = top_prob.squeeze(0).tolist()
    top_idx = top_idx.squeeze(0).tolist()

    # Map model indices -> class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in top_idx]

    # Optionally map to human-readable names
    if category_names is not None and os.path.isfile(category_names):
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name.get(c, c) for c in classes]

    return top_prob, classes


def main():
    parser = argparse.ArgumentParser(description='Predict image class with a trained model checkpoint')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (.pth)')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to JSON mapping of categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    args = parser.parse_args()

    probs, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)

    # Print results
    if args.top_k == 1:
        print(f"Prediction: {classes[0]} (p={probs[0]:.4f})")
    else:
        for rank, (c, p) in enumerate(zip(classes, probs), start=1):
            print(f"Top {rank}: {c} (p={p:.4f})")


if __name__ == '__main__':
    main()
