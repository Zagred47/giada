import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronioLoss(nn.Module):
    def __init__(self):
        super(NeuronioLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, model_output, target):
        # Extract the elements for the respective losses
        bce_input = model_output[..., 0]
        bce_target = target[0]
        mse_input = model_output[..., 1]
        mse_target = target[1]

        # Compute the losses
        bce_loss = self.bce_loss(bce_input, bce_target)
        mse_loss = self.mse_loss(mse_input, mse_target)

        # Balance the losses with a factor of 0.5 each
        loss = 0.5 * bce_loss + 0.5 * mse_loss

        return loss


class NeuronioRapidFireEPLLoss(nn.Module):
    """NeuronIO loss with rapid-fire-aware extreme penalization for soma.

    This keeps the original spike BCE term and replaces the soma MSE term with
    a tripartite loss inspired by EPL:
      - normal points: squared error
      - rapid-fire underestimation: exp(-error) - 1
      - rapid-fire overestimation: exp(error / lambda_over) - 1

    Thresholds must be provided in the training soma scale.
    """

    def __init__(
        self,
        high_threshold: float,
        derivative_threshold: float,
        lambda_over: float = 3.0,
        smooth_window: int = 7,
        spike_weight: float = 0.5,
        soma_weight: float = 0.5,
        exp_input_clip: float = 20.0,
    ):
        super(NeuronioRapidFireEPLLoss, self).__init__()

        if lambda_over <= 0:
            raise ValueError("lambda_over must be positive.")
        if smooth_window < 1:
            raise ValueError("smooth_window must be >= 1.")
        if smooth_window % 2 == 0:
            raise ValueError("smooth_window must be odd to preserve sequence length.")

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.high_threshold = high_threshold
        self.derivative_threshold = derivative_threshold
        self.lambda_over = lambda_over
        self.smooth_window = smooth_window
        self.spike_weight = spike_weight
        self.soma_weight = soma_weight
        self.exp_input_clip = exp_input_clip

    def forward(self, model_output, target):
        spike_logit = model_output[..., 0]
        soma_pred = model_output[..., 1]

        spike_target = target[0]
        soma_target = target[1]

        spike_loss = self.bce_loss(spike_logit, spike_target)
        soma_loss = self._rapid_fire_epl(soma_pred, soma_target)

        return self.spike_weight * spike_loss + self.soma_weight * soma_loss

    def _rapid_fire_epl(self, soma_pred, soma_target):
        error = soma_pred - soma_target
        rapid_mask = self._rapid_fire_mask(soma_target)

        normal_loss = error.pow(2)

        under_input = torch.clamp(-error, max=self.exp_input_clip)
        over_input = torch.clamp(error / self.lambda_over, max=self.exp_input_clip)

        under_extreme_loss = torch.exp(under_input) - 1.0
        over_extreme_loss = torch.exp(over_input) - 1.0

        extreme_loss = torch.where(
            error < 0,
            under_extreme_loss,
            over_extreme_loss,
        )
        loss = torch.where(rapid_mask, extreme_loss, normal_loss)

        return loss.mean()

    def _rapid_fire_mask(self, soma_target):
        smoothed = self._smooth_time(soma_target)
        derivative = torch.zeros_like(smoothed)
        derivative[:, 1:] = smoothed[:, 1:] - smoothed[:, :-1]

        return (soma_target >= self.high_threshold) & (
            derivative >= self.derivative_threshold
        )

    def _smooth_time(self, x):
        if self.smooth_window <= 1:
            return x

        kernel_size = self.smooth_window
        padding = kernel_size // 2
        x_3d = x.unsqueeze(1)
        smoothed = F.avg_pool1d(
            x_3d,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        return smoothed.squeeze(1)
