import torch
import torch.nn as nn
from typing import Union
from typing import Optional


class HiddenBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm_class=nn.Identity,
        activation_class=nn.ReLU,
        dropout_class=nn.Identity,
        use_skip: bool = False,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.in_features = in_features
        self.out_features = out_features

        self.fc = nn.Linear(in_features, out_features)
        self.norm = norm_class(out_features)

        self.projection = None
        if self.use_skip:
            if in_features != out_features:
                self.projection = nn.Linear(in_features, out_features)

        self.activation_fn = activation_class()
        self.dropout = dropout_class()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = self.norm(h)

        if self.use_skip:
            if self.projection is not None:
                skip_x = self.projection(x)
            else:
                skip_x = x
            out = h + skip_x
        else:
            out = h

        out = self.activation_fn(out)
        out = self.dropout(out)

        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_depth: int = 0,
        hidden_neurons: Union[list, int] = [],
        hidden_skip: bool = False,
        input_norm_class=nn.Identity,
        hidden_norm_class=nn.Identity,
        hidden_activation_class=nn.ReLU,
        dropout_class=nn.Identity,
        has_output_layer: bool = True,
        out_features: Optional[int] = None,
        output_activation_class=nn.Identity,
    ):
        super().__init__()
        self.input_norm_layer = input_norm_class(in_features)

        self.hidden_layers, last_hidden_out_features = self.create_hidden_layers(
            in_features,
            hidden_depth,
            hidden_neurons,
            hidden_skip,
            hidden_norm_class,
            hidden_activation_class,
            dropout_class,
        )

        if has_output_layer:
            assert (
                out_features is not None
            ), "out_features must be provided when has_output_layer is True!"
            self.output_layer = nn.Sequential(
                nn.Linear(
                    in_features=last_hidden_out_features, out_features=out_features
                ),
                output_activation_class(),
            )
        else:
            assert (
                hidden_depth > 0
            ), "hidden_depth must be greater than 0 when has_output_layer is False!"
            self.output_layer = nn.Sequential(nn.Identity(), output_activation_class())

    def create_hidden_layers(
        self,
        in_features: int,
        hidden_depth: int,
        neurons: Union[list, int],
        hidden_skip: bool,
        norm_class: nn.Module,
        activation_class: nn.Module,
        dropout_class: nn.Module,
    ) -> tuple:
        neurons = self.format_neurons(neurons, hidden_depth)

        if len(neurons) == 0:
            self.hidden_depth = hidden_depth
        else:
            self.hidden_depth = len(neurons)

        hidden_layers = []

        for hidden_layer_out_features in neurons:
            hidden_block = HiddenBlock(
                in_features=in_features,
                out_features=hidden_layer_out_features,
                norm_class=norm_class,
                activation_class=activation_class,
                dropout_class=dropout_class,
                use_skip=hidden_skip,
            )
            hidden_layers.append(hidden_block)
            in_features = hidden_layer_out_features

        hidden_layers = nn.ModuleList(hidden_layers)
        last_hidden_layer_out_features = in_features

        return hidden_layers, last_hidden_layer_out_features

    def format_neurons(self, neurons: Union[list, int], hidden_depth: int) -> list:
        if isinstance(neurons, int):
            assert hidden_depth > 0, "hidden_depth must be greater than 0 when neurons is an int!"
            neurons = [neurons] * hidden_depth

        return neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm_layer(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        x = self.output_layer(x)

        return x
