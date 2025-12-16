# predict.py

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config as config
from models.resunet import ResUNet

# --- 1. Settings ---
INPUT_DIR = 'test_data'
OUTPUT_DIR = 'test_results'
MODEL_PATH = '/home/tec/Desktop/Project/ResUNet-Seg/checkpoints/ResNet50_LaPa_Run1/best_model.pth'

# --- 2. LaPa Color Palette ---
PALETTE = np.array([
    [0, 0, 0],          # 0: Background
    [255, 204, 153],    # 1: Skin
    [0, 255, 0],        # 2: L Eyebrow
    [0, 255, 0],        # 3: R Eyebrow
    [0, 0, 255],        # 4: L Eye
    [0, 0, 255],        # 5: R Eye
    [255, 165, 0],      # 6: Nose
    [255, 0, 0],        # 7: U Lip
    [128, 0, 0],        # 8: I Mouth
    [255, 0, 0],        # 9: L Lip
    [128, 0, 128]       # 10: Hair
], dtype=np.uint8)

def process_image(img_path, model, device, transform):
    """
    讀取圖片 -> 預測 -> Resize 回原圖尺寸 -> 回傳 Mask
    """
    # 1. Load Image & Get Original Size
    original_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = original_img.size  # origin size (Width, Height)
    
    # 2. Preprocess (Resize to 512x512 for Model)
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor)
        # get 512x512 predict results
        pred_mask_512 = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
    # 4. Resize Mask back to Original Size
    # [重要] 必須使用 INTER_NEAREST (最近鄰插值)，才不會破壞類別整數 (例如把 1 和 3 平均成 2)
    pred_mask_orig = cv2.resize(pred_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
    return pred_mask_orig

def save_mask_only(pred_mask, save_path):
    """
    只儲存上色後的 Mask
    """
    # 1. Apply Color Map
    # pred_mask (H, W) -> color_mask (H, W, 3)
    color_mask = PALETTE[pred_mask]
    
    # 2. Save
    # OpenCV uses BGR, so convert RGB -> BGR
    cv2.imwrite(save_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input dir not found: {INPUT_DIR}")
        return

    device = config.DEVICE
    print(f"🚀 Loading Model: {MODEL_PATH}")
    
    model = ResUNet(n_classes=config.NUM_CLASSES).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    # Transform，must be Resize to 512
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_ext)]
    
    print(f"📂 Found {len(image_files)} images")
    
    for img_name in tqdm(image_files, desc="Processing"):
        img_path = os.path.join(INPUT_DIR, img_name)
        
        # Save as PNG
        save_name = os.path.splitext(img_name)[0] + ".png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        try:
            mask = process_image(img_path, model, device, transform)
            save_mask_only(mask, save_path)
            
        except Exception as e:
            print(f"⚠️ Error: {img_name} - {e}")

    print(f"\n✅ Done! Original-size masks saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()