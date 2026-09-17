import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    """
    SwiGLU activation.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the SwiGLU activation.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.nn.functional.silu(x2)

if __name__ == "__main__":
    # Example usage
    x = torch.randn(4, 8)  # Batch of 4, 8 features (must be divisible by 2)
    model = SwiGLU()
    output = model(x)
    print(output)