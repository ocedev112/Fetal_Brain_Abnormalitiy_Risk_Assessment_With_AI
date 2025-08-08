import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from src.effnet_ssa import EffNetB3_SSA

class FU_model_wrapper:
    def __init__(self, model_path="models/fu_model.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EffNetB3_SSA(num_classes=5).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.class_names = ["abdomen", "brain", "femur", "other", "thorax"]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]) 
        ])

    def validate_image(self, image):
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            else:
                image = image.convert("RGB")
        except Exception as e:
            return {"error": f"invalid Image {e}", "is_brain": False}
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_index = torch.argmax(probs, dim=1).item()
            pred_class = self.class_names[pred_index]
            confidence = probs[0, pred_index].item()
        
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "is_brain": pred_class == "brain"
        }
