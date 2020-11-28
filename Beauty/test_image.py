from models import *
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
model = se_resnet_face50(out_planes=5)
checkpoint = torch.load("./weight_local/beauty_epoch_50.pth", map_location='cpu')
# checkpoint = torch.load("./weight_local/beauty_ft_epoch_50.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

preproc = transforms.Compose([transforms.Resize(224),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])

with torch.no_grad():
    img = Image.open("/Users/hulk/Documents/SCUT_FBP5500_Asian/images_fine/AF3.jpg")
    image = preproc(img).unsqueeze(0)

    outputs = model(image)
    logit = torch.softmax(outputs, dim=1).squeeze().numpy()
    score = np.sum(logit * [1, 2, 3, 4, 5])
    print(logit)
    print(score)
