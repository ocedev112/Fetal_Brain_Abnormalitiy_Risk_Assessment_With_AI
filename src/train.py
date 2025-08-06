import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets
from torch.utils.data import DataLoader
from utils import get_transforms, calculate_accuracy
from effnet_ssa import EffNetB3_SSA


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = datasets.ImageFolder(f"Fetal-Brain-Abnormalities-Ultrasound-1/train", transform=get_transforms())
val_dataset = datasets.ImageFolder(f"Fetal-Brain-Abnormalities-Ultrasound-1/valid", transform=get_transforms())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)


model = EffNetB3_SSA(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2)

EPOCHS = 25
train_losses, val_losses, val_acc = [],[],[]

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0,0,0
    for images, labels in train_loader:
        images, labels = images.to(device),  labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
       
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct+=(preds==labels).sum().item()
        total+=labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={total_loss:.3f} Acc={train_acc:.3f}")
   

    model.eval()
    val_correct, val_total, val_loss= 0,0,0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total +=labels.size(0)
        

    val_acc = val_correct / val_total
    scheduler.step(val_loss)


    accuracy = correct / len(val_dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss={val_loss:.3f} Acc={val_acc:.3f}")
 


torch.save(model.state_dict(), "models/model.pth")
print("Model saved")