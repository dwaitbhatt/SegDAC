import torch
import torch.nn as nn
from typing import List, Optional, Type

class ResidualBlock(nn.Module):
    """
    Residual Block using the Pre-activation style (Norm -> Act -> Linear -> Add).
    Matches ResNet v2 structure for potentially better gradient flow.
    Output = Linear(Activation(Norm(x))) + Projection(x)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm_class: Type[nn.Module] = nn.LayerNorm,
        activation_class: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.norm = norm_class(in_features)
        self.activation = activation_class()
        
        self.fc = nn.Linear(in_features, out_features)

        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features, bias=False)
        else:
            self.projection = nn.Identity() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.projection(x)

        h = self.norm(x)
        h = self.activation(h)
        h = self.fc(h)

        out = h + skip
        return out


class ResidualMLP(nn.Module):
    """
    Residual Multi-Layer Perceptron using pre-activation ResidualBlocks.
    """
    def __init__(
        self,
        in_features: int,
        hidden_neurons: List[int],
        out_features: Optional[int] = None,
        norm_class: Type[nn.Module] = nn.LayerNorm,
        activation_class: Type[nn.Module] = nn.ReLU,
        use_input_norm: bool = True,
        use_final_norm: bool = True,
        use_final_act: bool = True
    ):
        super().__init__()
        
        self.input_norm = norm_class(in_features) if use_input_norm else nn.Identity()

        self.blocks = nn.ModuleList()
        current_dim = in_features
        for h_dim in hidden_neurons:
            block = ResidualBlock(
                in_features=current_dim,
                out_features=h_dim,
                norm_class=norm_class,
                activation_class=activation_class,
            )
            self.blocks.append(block)
            current_dim = h_dim

        if use_final_norm:
            self.final_norm = norm_class(current_dim) 
        else:
            self.final_norm = nn.Identity()
        
        if use_final_act:
            self.final_activation = activation_class()
        else:
            self.final_activation = nn.Identity()

        if out_features is not None:
            self.output_layer = nn.Linear(current_dim, out_features)
        else:
            self.output_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = self.final_activation(x)
            
        x = self.output_layer(x)
            
        return x