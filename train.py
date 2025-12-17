# train.py

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# --- Project Setup ---
# Ensure we can import from src/ and project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Imports ---
import config as config
from models.resunet import ResUNet
from utils.dataset import MultiClassSegDataset
from utils.logger import CSVLogger
from utils.plot import plot_history
from utils.loss import SegmentationLoss
from utils.metrics import calculate_dice

# --- 1. Helper Functions for Setup ---
def get_loaders():
    """Initializes and returns Train/Val DataLoaders."""
    print(f"📂 Train Path: {config.TRAIN_DIR}")
    print(f"📂 Val Path:   {config.VAL_DIR}")

    train_ds = MultiClassSegDataset(
        root_dir=config.TRAIN_DIR,
        img_folder=config.IMG_DIR,
        mask_folder=config.MASK_DIR,
        mode='train',
        img_size=config.IMG_SIZE
    )
    
    val_ds = MultiClassSegDataset(
        root_dir=config.VAL_DIR,
        img_folder=config.IMG_DIR,
        mask_folder=config.MASK_DIR,
        mode='val',
        img_size=config.IMG_SIZE
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    print(f"✅ Data Loaded: Train={len(train_ds)}, Val={len(val_ds)}")
    return train_loader, val_loader

def get_model_components(device):
    """Initializes Model, Loss, Optimizer, Scaler, and Scheduler."""
    model = ResUNet(n_classes=config.NUM_CLASSES).to(device)
    
    loss_fn = SegmentationLoss(
        n_classes=config.NUM_CLASSES, 
        ignore_index=config.IGNORE_INDEX
    )
    
    # Use AdamW Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2)
    
    scaler = torch.amp.GradScaler()
    
    # Use Cosine Annealing with Warm Restarts Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    return model, loss_fn, optimizer, scaler, scheduler

# --- 2. Core Loops ---
def run_epoch(loader, model, optimizer, loss_fn, scaler, device, mode='train'):
    """
    Unified function for both Training and Validation loops.
    mode: 'train' or 'val' 
    """
    model.train() if mode == 'train' else model.eval()
    loop = tqdm(loader, desc=mode.capitalize(), leave=False)
    
    total_loss = 0.0
    total_dice = 0.0
    
    # Enable gradient calculation only for training
    with torch.set_grad_enabled(mode == 'train'):
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            # Forward Pass (AMP)
            with torch.amp.autocast('cuda'):
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            # Backward Pass (Only for Train)
            if mode == 'train':
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Metrics (Detach to save memory)
            dice = calculate_dice(
                predictions.detach(), 
                targets, 
                n_classes=config.NUM_CLASSES, 
                ignore_index=config.IGNORE_INDEX
            )
            
            total_loss += loss.item()
            total_dice += dice
            loop.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice:.4f}")
            
    return total_loss / len(loader), total_dice / len(loader)

# --- 3. Main Execution ---
def main():
    print(f"--- 🚀 Starting Training on {config.DEVICE} ---")
    
    # A. Setup
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    logger = CSVLogger(save_dir=config.CHECKPOINT_DIR, filename='training_log.csv')
    
    # B. Load Data & Model
    train_loader, val_loader = get_loaders()
    model, loss_fn, optimizer, scaler, scheduler = get_model_components(config.DEVICE)

    # C. Training Loop
    best_dice = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # 1. Train & Validate
        train_loss, train_dice = run_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, mode='train')
        val_loss, val_dice = run_epoch(val_loader, model, None, loss_fn, None, config.DEVICE, mode='val')
        
        # [Modified] Update Scheduler
        # Unlike ReduceLROnPlateau, CosineAnnealingWarmRestarts does not need val_loss.
        # It updates based on the epoch count.
        scheduler.step()

        # 2. Logging
        current_lr = optimizer.param_groups[0]['lr']
        logger.log([epoch+1, current_lr, train_loss, train_dice, val_loss, val_dice])
        print(f"\tTrain Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"\tVal Loss:   {val_loss:.4f} | Dice: {val_dice:.4f}")
        
        # 3. Save History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

        # 4. Save Checkpoints
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best Model Saved! (Dice: {best_dice:.4f})")
            
        # 5. Save the latest Checkpoints
        torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"last_model.pth"))

    # D. Finalize
    print("\n🎉 Training Complete!")
    plot_history(
        history['train_loss'], history['val_loss'], 
        history['train_dice'], history['val_dice'], 
        save_dir=config.CHECKPOINT_DIR # Save plot in checkpoints folder
    )
    print(f"📈 Results saved to {config.CHECKPOINT_DIR}")

if __name__ == "__main__":
    main()