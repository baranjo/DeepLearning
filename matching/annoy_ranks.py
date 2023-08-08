
import os

from PIL import Image
import torch.nn as nn
from torchvision import models, transforms
from annoy import AnnoyIndex


images_folder = "images/"
images = os.listdir(images_folder)

weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=weights)
model.fc = nn.Identity()

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

annoy_index = AnnoyIndex(512, 'angular')

for i in range(len(images)):
    image = Image.open(os.path.join(images_folder, images[i]))
    input_tensor = transform(image).unsqueeze(0)

    if input_tensor.size()[1] == 3:
        output_tensor = model(input_tensor)
        annoy_index.add_item(i, output_tensor[0])

        if i % 100 == 0:
            print(f'Processed { i } images.')

annoy_index.build(10)
annoy_index.save('bird_index.ann')
