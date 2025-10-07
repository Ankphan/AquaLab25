import torch
from torchmetrics import MeanSquaredError

def wdro_robust_loss(model, inputs, epsilon=0.1, device='cpu'):
    """
    Compute the Wasserstein Distributionally Robust Optimization (WDRO) loss.

    Args:
        model (torch.nn.Module): The model being trained.
        batch (tuple): A batch from your DataLoader, expected to be (inputs, targets).
        epsilon (float): The radius of the Wasserstein ball, representing perturbation strength.
        device (str): The computation device ('cpu' or 'gpu').

    Returns:
        torch.Tensor: The computed robust loss value.
    """
    # Unpack the batch
    inputs = inputs.to(device)

    # Standard MSE Loss for the original inputs
    mse_loss = MeanSquaredError().to(device)
    original_output = model(inputs)
    loss_original = mse_loss(original_output, inputs)

    # Perturb inputs within a Wasserstein ball around the original inputs
    noise = torch.randn_like(inputs) * epsilon
    perturbed_inputs = inputs + noise
    perturbed_output = model(perturbed_inputs)
    loss_perturbed = mse_loss(perturbed_output, inputs)

    # Combine the original loss and the perturbed loss
    robust_loss = torch.max(loss_original, loss_perturbed)

    return robust_loss