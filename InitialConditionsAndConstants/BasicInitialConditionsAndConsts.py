import torch  # Use GPU tensors for initial condition generation to avoid NumPy CPU overhead

def generate_initial_temps(x, y) -> list[int]: # A list of temperatures at each sample point
    '''
    Given two nparrays, return the initial temp - pretty simple, going off the diagram on page 171 of the paper
    '''
    chamber_width, chamber_height = 20000, 6000 # Same for chamber size - normalize to 0 - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Keep computations on GPU when available
    x_flat = torch.tensor(x, dtype=torch.float32, device=device).flatten()  # Convert x samples to GPU tensor for faster math
    y_flat = torch.tensor(y, dtype=torch.float32, device=device).flatten()  # Convert y samples to GPU tensor for faster math

    mask = (x_flat > (6000 / chamber_width)) | (y_flat > (3000 / chamber_height))  # GPU mask replicating piecewise region check
    upper_region = 170.0 - 25.0 * (y_flat / (1000.0 / chamber_height))  # GPU vectorized computation for upper region temps
    lower_region = 900.0 - (700.0 * y_flat) - (700.0 * x_flat)  # GPU vectorized computation for lower region temps
    temps = torch.where(mask, upper_region, lower_region)  # Branch without CPU loop while matching original logic

    return temps.detach().cpu().tolist()  # Return CPU list so existing callers keep the same interface
