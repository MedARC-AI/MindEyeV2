# import torch
# import os
# from torchvision.models import inception_v3
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchvision.transforms import functional as TF
# from PIL import Image
# from tqdm import tqdm

# device = torch.device("cuda")

# # Function to load and preprocess images from a directory
# def load_images_from_directory(directory, expected_count=30000):
#     images = []
#     image_count = 0

#     for filename in tqdm(os.listdir(directory), desc=f"Loading images from {directory}"):
#         if filename.endswith('.jpg') or filename.endswith('.png'):  # Check for image files
#             image_count += 1
#             img = Image.open(os.path.join(directory, filename)).convert('RGB')
#             img = TF.resize(img, [299, 299])  # Resize image
#             img = TF.to_tensor(img)  # Convert to tensor
#             img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
#             images.append(img.unsqueeze(0))  # Add batch dimension

#     if image_count != expected_count:
#         raise ValueError(f"Expected {expected_count} images in directory '{directory}', but found {image_count} images.")

#     return torch.cat(images, dim=0)  # Concatenate all images into a single tensor

# # Load the Inception V3 model
# model = inception_v3(pretrained=True)
# model.fc = torch.nn.Identity()  # Remove the final fully connected layer
# model.eval().to(device)

# # Directories containing your images
# real_images_dir = '/weka/proj-fmri/shared/coco/sampled_imgs'
# generated_images_dir = '/weka/proj-fmri/shared/coco/vd_imgs'

# # Load images
# real_images = load_images_from_directory(real_images_dir).to(device)
# generated_images = load_images_from_directory(generated_images_dir).to(device)

# # Extract features
# with torch.no_grad():  # Disable gradient calculation for efficiency
#     real_features = model(real_images)
#     generated_features = model(generated_images)

# # Initialize FID
# fid = FrechetInceptionDistance(feature=2048)

# # Update state with real and generated features
# fid.update(real_features, real=True)
# fid.update(generated_features, real=False)

# # Compute FID score
# fid_score = fid.compute()
# print(f'FID score: {fid_score}')

# import torch
# import os
# from torchvision.models import inception_v3
# from torchmetrics.image.fid import FrechetInceptionDistance
# from torchvision.transforms import functional as TF
# from torchvision.transforms import ToTensor, Resize
# from PIL import Image
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm

# # Detect if we have a GPU available and set up Data Parallelism
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs!")

# # Custom dataset class
# class ImageDataset(Dataset):
#     def __init__(self, directory):
#         self.directory = directory
#         self.filenames = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.directory, self.filenames[idx])).convert('RGB')
#         img = Resize((299, 299))(img)
#         img = ToTensor()(img)  # Converts to [0, 1] range
#         img = (img * 255).to(torch.uint8)  # Scales to [0, 255] and converts to uint8
#         return img

# # Load the Inception V3 model and apply Data Parallelism
# model = inception_v3(pretrained=True)
# model.fc = torch.nn.Identity()
# model = model.to(device)
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)

# model.eval()

# # Load datasets
# real_images_dataset = ImageDataset('/weka/proj-fmri/shared/coco/sampled_imgs')
# generated_images_dataset = ImageDataset('/weka/proj-fmri/shared/coco/vd_imgs')

# # Create dataloaders
# batch_size = 30  # Adjust batch size as needed
# real_images_loader = DataLoader(real_images_dataset, batch_size=batch_size, shuffle=False)
# generated_images_loader = DataLoader(generated_images_dataset, batch_size=batch_size, shuffle=False)

# # Function to process images through model
# def extract_features(loader):
#     features_list = []
#     with torch.no_grad():
#         for images in tqdm(loader, desc="Processing images"):
#             images = images.to(device)
#             features = model(images)
#             features_list.append(features.cpu())  # Move features to CPU to avoid GPU memory overload
#     return torch.cat(features_list, dim=0)

# # Process real and generated images and extract features
# real_features = extract_features(real_images_loader)
# generated_features = extract_features(generated_images_loader)

# # Initialize FID with the correct feature size
# fid = FrechetInceptionDistance(feature=2048)  # Adjust feature size if necessary

# # Update FID with extracted features
# fid.update(real_features, real=True)
# fid.update(generated_features, real=False)

# # Compute FID score
# fid_score = fid.compute()
# print(f'FID score: {fid_score}')


# # Compute FID score
# fid_score = fid.compute()
# print(f'FID score: {fid_score}')

import torch
import os
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Detect if we have a GPU available and set up Data Parallelism
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")

# Custom dataset class, modified to load only up to 1000 images
class ImageDataset(Dataset):
    def __init__(self, directory, max_images=30000):
        self.directory = directory
        self.filenames = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
        self.filenames = self.filenames[:max_images]  # Limit to first 1000 images

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.directory, self.filenames[idx]))
        img = Resize((512, 512))(img)
        img = ToTensor()(img)  # Converts to [0, 1] range
        img = (img * 255).to(torch.uint8)  # Scales to [0, 255] and converts to uint8
        return img

# Load datasets
real_images_dataset = ImageDataset('/weka/proj-fmri/shared/coco/sampled_imgs')
generated_images_dataset = ImageDataset('/weka/proj-fmri/shared/coco/vd_imgs')


# Create dataloaders
batch_size = 10  # Adjust batch size as needed
real_images_loader = DataLoader(real_images_dataset, batch_size=batch_size, shuffle=False)
generated_images_loader = DataLoader(generated_images_dataset, batch_size=batch_size, shuffle=False)

# Function to process images through FID metric
def process_images(loader, fid, real):
    with torch.no_grad():
        for images in tqdm(loader, desc="Processing images"):
            images = images.to(device)
            fid.update(images, real=real)

# Initialize FID
fid = FrechetInceptionDistance(feature=2048).to(device)

# Process real and generated images
process_images(real_images_loader, fid, real=True)
process_images(generated_images_loader, fid, real=False)

# Compute FID score
fid_score = fid.compute()
print(f'FID score: {fid_score}')

