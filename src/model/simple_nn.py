from torch import nn

class SimpleNN(nn.Module):
    """Fully-connected neural network model with ReLU activations"""
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int = None
    ) -> None:
        """Initialize method for SimpleNN.
        
        Args:
        ----
            input_size (int): Input size
            hidden_sizes (list[int]): Hidden sizes (if output_size is None then the last size is output size)
            output_size (int): Output
         """
        super().__init__()
        layers = nn.ModuleList([])
        prev_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
        if output_size is not None:
            layers.append(nn.Linear(prev_size, output_size))
        self.layers = layers
    
    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = nn.functional.relu(out)
            out = layer(out)
        return out