import argparse
from typing import List
from pathlib import Path


import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_tfm, eval_tfm


def get_data_loaders(data_dir: str, batch_size: int, device: torch.device):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_tfm, eval_tfm = get_transforms()

    train_data = datasets.ImageFolder(train_dir, transform=train_tfm)
    valid_data = datasets.ImageFolder(valid_dir, transform=eval_tfm)
    test_data = datasets.ImageFolder(test_dir, transform=eval_tfm)

    pin = (device.type == 'cuda')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=pin)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=pin)

    return train_data, valid_data, test_data, train_loader, valid_loader, test_loader


def build_model(arch, hidden_units: List[int], num_classes):
    arch = arch.lower()
    if arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        in_features = model.fc.in_features
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
        classifier_info = {
            'type': 'fc', 'in_features': in_features, 'hidden_units': hidden_units, 'num_classes': num_classes
        }
        return model, classifier_info

    elif arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for p in model.parameters():
            p.requires_grad = False
        # vgg16 classifier first linear in_features
        if hasattr(model.classifier[0], 'in_features'):
            in_features = model.classifier[0].in_features
        else:
            # Fallback size for VGG16 (usually 25088)
            in_features = 25088
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
        classifier_info = {
            'type': 'classifier', 'in_features': in_features, 'hidden_units': hidden_units, 'num_classes': num_classes
        }
        return model, classifier_info

    else:
        raise ValueError(f"Unsupported architecture: {arch}. Use 'resnet18' or 'vgg16'.")


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logps = model(images)
            loss = criterion(logps, labels)
            val_loss += loss.item() * images.size(0)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            correct += (top_class.squeeze() == labels).type(torch.float).sum().item()
            total += labels.size(0)
    avg_loss = val_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return avg_loss, acc


def train(args):
    device = torch.device('cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu')

    train_data, valid_data, test_data, train_loader, valid_loader, test_loader = get_data_loaders(
        args.data_dir, args.batch_size, device
    )
    num_classes = len(train_data.classes)

    model, classifier_info = build_model(args.arch, args.hidden_units, num_classes)

    # Attach mapping for later use
    model.class_to_idx = train_data.class_to_idx

    # Only train classifier parameters
    if args.arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    criterion = nn.NLLLoss()
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_train = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_train += batch_size

        train_loss = running_loss / total_train if total_train > 0 else 0.0
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

        print(f"Epoch: {epoch+1}/{args.epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.2f}%")

    # Optional: test evaluation at end
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Save checkpoint
    #save_dir = args.save_dir or '.'
    #os.makedirs(save_dir, exist_ok=True)
    #ckpt_path = os.path.join(save_dir, f"{args.arch}_checkpoint.pth")
    
    save_dir = Path(args.save_dir or '.')
    save_dir.mkdir(exist_ok=True)
    ckpt_path = save_dir / f"{args.arch}_checkpoint.pth"


    checkpoint = {
        'arch': args.arch,
        'classifier_info': classifier_info,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train an image classifier.')
    parser.add_argument('data_dir', type=str, help='Path to dataset root containing train/valid/test folders')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'vgg16'])
    parser.add_argument('--hidden_units', type=int, default=[256, 128])
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
