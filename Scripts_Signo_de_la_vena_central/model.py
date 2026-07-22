import torch
import torch.nn as nn


class CVSNet(nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        def make_branch(in_channels: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2, stride=2),

                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=0.2),
            )

        self.flair_branch = make_branch()
        self.swi_branch = make_branch()
        self.fusion = nn.Sequential(
            nn.Conv3d(16 + 16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flair = x[:, 0:1, :, :, :]
        swi = x[:, 1:2, :, :, :]

        # Cada modalidad pasa por su propia rama convolucional.
        flair_features = self.flair_branch(flair)  
        swi_features = self.swi_branch(swi)         


        # aprende a combinar ambas modalidades.
        fused = torch.cat([flair_features, swi_features], dim=1)  
        fused = self.fusion(fused)                                 

        # Global Average Pooling 
        pooled = self.global_pool(fused)        
        pooled = pooled.view(pooled.size(0), -1)  

        # clasificador final.
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled) 

        return logits


if __name__ == "__main__":
    x = torch.randn(2, 2, 28, 28, 28)

    model = CVSNet()
    output = model(x)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Parámetros entrenables:", n_params)
