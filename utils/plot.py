# utils/plot.py

import matplotlib.pyplot as plt
import os

def plot_history(train_losses, val_losses, train_scores, val_scores, save_dir):
    """
    Plots the training/validation loss and dice scores.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 1. Plot Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Plot Dice Score Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_scores, 'b-', label='Train Dice')
    plt.plot(epochs, val_scores, 'g-', label='Val Dice')
    plt.title('Dice Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    # Save plot
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"📊 Training curves saved at: {save_path}")