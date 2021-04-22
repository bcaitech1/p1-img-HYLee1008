def train(model, train_data_loader, optimizer, criterion, device):
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in train_data_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return correct, total
        
    