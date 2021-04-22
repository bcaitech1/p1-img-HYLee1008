import torch


def inference(model, valid_loader, device):
     # evaluate on validation set
    correct = 0
    total = 0

    model.eval()
    for images, labels in valid_loader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return correct, total