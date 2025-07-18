from PIL import Image
import torch

import torch.nn as nn
from torchvision import transforms,models

trained_model = None
classes = ['Front Breakage','Front Crushed','Front Normal','Rear Breakage','Rear Crushed','Rear Normal']

class ResnetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.model.fc.in_features,6)
        )

    def forward(self,x):
        x = self.model(x)
        return x


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transformation = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

    tensor_image = transformation(image).unsqueeze(0) #Addding dimension as model is trained in batches
    
    global trained_model 
    
    if trained_model is None:
        trained_model = ResnetClassifier()
        trained_model.load_state_dict(torch.load("model\saved_model.pth"))
        trained_model.eval()

    with torch.no_grad():
        out = trained_model(tensor_image)
        _,predicted_class  = torch.max(out,1)



    return classes[predicted_class.item()]