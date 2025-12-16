# models/resunet.py

import torch
import torch.nn as nn
import torchvision.models as models
from models.blocks import Up, OutConv 

class ResUNet(nn.Module):
    def __init__(self, n_classes=1):
        super(ResUNet, self).__init__()
        self.n_classes = n_classes

        # -----------------------------------------------------------------
        # 1. Backbone (Encoder): ResNet50
        # -----------------------------------------------------------------
        # Use pretrained weights from ImageNet
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Extract layers for Skip Connections
        # ResNet50 Channel counts:
        # layer0 (conv1) -> 64
        # layer1         -> 256
        # layer2         -> 512
        # layer3         -> 1024
        # layer4         -> 2048
        
        self.base_layers = list(base_model.children())
        
        # Layer 0: Input (3, H, W) -> Conv1 (64, H/2, W/2)
        # We separate the pooling layer to use the pre-pool feature map for skip connection
        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer0_pool = self.base_layers[3]  # MaxPool (H/4, W/4)

        self.layer1 = base_model.layer1 # -> (256, H/4, W/4)
        self.layer2 = base_model.layer2 # -> (512, H/8, W/8)
        self.layer3 = base_model.layer3 # -> (1024, H/16, W/16)
        self.layer4 = base_model.layer4 # -> (2048, H/32, W/32)

        # -----------------------------------------------------------------
        # 2. Decoder (Up-sampling)
        # -----------------------------------------------------------------
        # We need to calculate input channels carefully.
        # Logic: in_channels = (Channels from Previous Up Block) + (Skip Connection Channels)
        
        # UpBlock 1: 
        # Input: Layer4 (2048) -> Upsampled
        # Skip:  Layer3 (1024)
        # Total input channels for DoubleConv = 2048 + 1024 = 3072
        self.up1 = Up(2048 + 1024, 1024, bilinear=True)

        # UpBlock 2: 
        # Input: Up1 Output (1024) -> Upsampled
        # Skip:  Layer2 (512)
        # Total input channels = 1024 + 512 = 1536
        self.up2 = Up(1024 + 512, 512, bilinear=True)

        # UpBlock 3: 
        # Input: Up2 Output (512) -> Upsampled
        # Skip:  Layer1 (256)
        # Total input channels = 512 + 256 = 768
        self.up3 = Up(512 + 256, 256, bilinear=True)

        # UpBlock 4: 
        # Input: Up3 Output (256) -> Upsampled
        # Skip:  Layer0 (64) - This is the output of the first Conv1
        # Total input channels = 256 + 64 = 320
        self.up4 = Up(256 + 64, 64, bilinear=True)

        # -----------------------------------------------------------------
        # 3. Final Output
        # -----------------------------------------------------------------
        # At this point, the tensor size is (H/2, W/2) because Layer0 was stride 2.
        # We need one last upsample to match original input resolution (H, W).
        
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # --- Encoder Path (ResNet) ---
        x0 = self.layer0(x)         # Output: (64, H/2, W/2) -> Skip for up4
        x0_p = self.layer0_pool(x0) # Output: (64, H/4, W/4)
        
        x1 = self.layer1(x0_p)      # Output: (256, H/4, W/4) -> Skip for up3
        x2 = self.layer2(x1)        # Output: (512, H/8, W/8) -> Skip for up2
        x3 = self.layer3(x2)        # Output: (1024, H/16, W/16) -> Skip for up1
        x4 = self.layer4(x3)        # Output: (2048, H/32, W/32) -> Bridge

        # --- Decoder Path (U-Net) ---
        # Up(current_features, skip_features)
        
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0) 
        
        # Final upsampling to restore original image size
        d_final = self.final_upsample(d4)
        
        # Output Logits
        logits = self.outc(d_final)
        
        return logits
    
    def predict(self, x):
        """
        Helper function for inference.
        Returns:
            - Binary Mask (0 or 1) if n_classes == 1
            - Class Indices if n_classes > 1
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.n_classes == 1:
                # Binary Classification: Sigmoid + Threshold
                return (torch.sigmoid(logits) > 0.5).float()
            else:
                # Multi-class: Softmax + Argmax
                probs = torch.softmax(logits, dim=1)
                return torch.argmax(probs, dim=1)