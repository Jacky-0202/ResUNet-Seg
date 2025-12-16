# utils/dataset.py

import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class BinarySegDataset(Dataset):
    def __init__(self, root_dir, img_folder='im', mask_folder='gt', mode='train', img_size=320):
        """
        Generic Dataset class for Binary Segmentation with Data Augmentation.
        
        Args:
            root_dir (str): Root directory of the dataset.
            img_folder (str): Name of the folder containing images.
            mask_folder (str): Name of the folder containing masks.
            mode (str): 'train' or 'test'. Augmentation is applied only in 'train' mode.
            img_size (int): Target size for resizing images and masks.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        # 1. Construct full paths
        self.image_dir = os.path.join(root_dir, img_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Directories not found. Please check paths:\nImg: {self.image_dir}\nMask: {self.mask_dir}")

        # 2. Get all image filenames
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_ext)])
        
        if len(self.image_list) == 0:
            print(f"⚠️ Warning: No images found in {self.image_dir}!")

        # 3. Define Base Transforms (Resize & Normalization)
        # Note: We don't put random transforms here because we need to sync Image/Mask manually.
        
        # Image Normalization (ImageNet standards)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Resize allows us to handle input of any size
        self.resize = transforms.Resize((self.img_size, self.img_size))
        self.resize_mask = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 1. Get image filename
        img_name = self.image_list[idx]
        
        # 2. Handle corresponding Mask filename (Force .png extension)
        file_prefix = os.path.splitext(img_name)[0]
        mask_name = file_prefix + '.png'
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 3. Load Image and Mask
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L') # L = Grayscale
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Mask not found for image: {mask_path}")
        
        # 4. Data Augmentation (Only for Training)
        # SOTA Technique: Random Geometric Transformations
        if self.mode == 'train':
            # Random Horizontal Flip (50% prob)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # Random Vertical Flip (50% prob)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random Rotation (-10 to 10 degrees)
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        # 5. Apply Resizing and ToTensor
        # Image: Resize -> ToTensor -> Normalize
        image = self.resize(image)
        image = TF.to_tensor(image)
        image = self.norm(image)
        
        # Mask: Resize -> ToTensor
        mask = self.resize_mask(mask)
        mask = TF.to_tensor(mask)
        
        # 6. Binarize Mask (0.0 / 1.0)
        mask[mask > 0.5] = 1.0
        mask[mask <= 0.5] = 0.0
        
        return image, mask
    
    
class MultiClassSegDataset(Dataset):
    def __init__(self, root_dir, img_folder='images', mask_folder='masks', mode='train', img_size=320):
        """
        Standard Multi-class Dataset for clean integer masks.
        
        Args:
            root_dir (str): Dataset root.
            img_folder (str): Image folder name.
            mask_folder (str): Mask folder name.
            mode (str): 'train' or 'val'.
            img_size (int): Target size.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        self.image_dir = os.path.join(root_dir, img_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)
        
        # Check directories
        if not os.path.exists(self.image_dir) or not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"❌ Directories not found: {self.image_dir} or {self.mask_dir}")

        # Filter: Only load images that have a corresponding mask
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        
        # Get mask filenames without extension for matching
        available_masks = set(os.path.splitext(f)[0] for f in os.listdir(self.mask_dir) if not f.startswith('.'))
        
        self.image_list = []
        for f in sorted(os.listdir(self.image_dir)):
            if f.lower().endswith(valid_ext):
                file_id = os.path.splitext(f)[0]
                if file_id in available_masks:
                    self.image_list.append(f)
        
        print(f"Dataset ({mode}): Found {len(self.image_list)} valid pairs.")

        # Transforms
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize_img = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.resize_mask = transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        file_id = os.path.splitext(img_name)[0]
        
        # Find mask extension (could be .png, .tif, etc.)
        # Here we assume .png, but you can adjust if dataset uses .tif
        mask_name = file_id + '.png' 
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
            # Open mask directly. PIL usually handles index/grayscale automatically.
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Error loading: {img_path} or {mask_path}")
            raise e
        
        # Augmentation
        if self.mode == 'train':
            # Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Random Rotation
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # Resize
        image = self.resize_img(image)
        mask = self.resize_mask(mask)

        # To Tensor
        image = TF.to_tensor(image)
        image = self.norm(image)
        
        # Convert Mask to LongTensor (Integers)
        mask_np = np.array(mask)
        mask_tensor = torch.from_numpy(mask_np).long()
        
        return image, mask_tensor