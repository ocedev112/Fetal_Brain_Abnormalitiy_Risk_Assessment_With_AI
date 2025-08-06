import torch

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from sklearn.metrics import classification_report

from effnet_ssa import EffNetB3_SSA



transform = transforms.Compose([

    transforms.Resize((300, 300)),

    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

])



test_data = datasets.ImageFolder("Fetal-Brain-classification-1/test", transform=transform)

test_loader = DataLoader(test_data, batch_size=16)


print("Classes in your dataset:")
for i, class_name in enumerate(test_data.classes):
    print(f"{i}: {class_name}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EffNetB3_SSA(num_classes=len(test_data.classes)).to(device)

model.load_state_dict(torch.load("model.pth", map_location=device), strict=False)

model.eval()



all_preds, all_labels = [], []



with torch.no_grad():

    for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())

        all_labels.extend(labels.cpu().numpy())



print(classification_report(all_labels, all_preds, target_names=test_data.classes))
print("Model saved")