import torch


class TanhEnvActionScaler:
    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        if len(action_low.shape) == 2:
            action_low = action_low[0]
        if len(action_high.shape) == 2:
            action_high = action_high[0]

        self.action_low = action_low
        self.action_high = action_high

    def scale(self, unscaled_action: torch.Tensor) -> torch.Tensor:
        "Scales Tanh action to the env action space."
        return (
            0.5 * (unscaled_action + 1) * (self.action_high - self.action_low)
            + self.action_low
        )

    def unscale(self, scaled_action: torch.Tensor) -> torch.Tensor:
        "Unscales env action to the Tanh action space."
        device = scaled_action.device
        action_low = self.action_low.to(device)
        action_high = self.action_high.to(device)
        return (2 * (scaled_action - action_low) / (action_high - action_low)) - 1


class IdentityEnvActionScaler:
    def scale(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def unscale(self, action: torch.Tensor) -> torch.Tensor:
        return action


class MultiBinaryEnvActionScaler:
    def __init__(self, nb_binary_actions: int, device: str):
        self.nb_binary_actions = nb_binary_actions
        self.total_nb_actions = 2 ** nb_binary_actions
        self.mask = 2**torch.arange(nb_binary_actions).sort(descending=True)[0].to(device=device)
        self.action_mapping = torch.stack([(torch.tensor([multi_action_index], device=device).bitwise_and(self.mask).ne(0).byte()) for multi_action_index  in range(self.total_nb_actions)])

    def scale(self, action: torch.Tensor) -> torch.Tensor:
        return self.action_mapping[action]

    def unscale(self, action: torch.Tensor) -> torch.Tensor:
        return (action * self.mask.to(action.device)).sum(dim=-1)
