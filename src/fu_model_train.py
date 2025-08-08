import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from effnet_ssa import EffNetB3_SSA

device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]) 
])

train_dataset = ImageFolder(f"FU-LoRA/data/training_lora", transform = transforms)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle= True)

class_names = train_dataset.classes
model = EffNetB3_SSA(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct +=predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    accuracy = 100. * correct/total
    print(f"{epoch+1}/{EPOCHS} Loss: {running_loss/len(train_loader):.4f} | Train Acc: {accuracy:.2f}%")
    scheduler.step(running_loss)



torch.save(model.state_dict(), "models/fu_model.pth")
print("Model saved")