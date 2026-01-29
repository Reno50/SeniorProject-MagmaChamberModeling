import torch
from torchviz import make_dot
from pathlib import Path
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.key import Key

# Get absolute path to workspace
WORKSPACE = Path(__file__).parent.absolute()

# Recreate the same network architecture manually
# Kinda jank, replicates the config because we can't easily import it here
network = FullyConnectedArch(
    input_keys=[Key("time"), Key("x"), Key("y")],
    output_keys=[Key("Temperature"), Key("XVelocity"), Key("YVelocity"), 
                 Key("Pressure_water"), Key("Pressure_steam"), 
                 Key("Saturation_water"), Key("Saturation_steam")],
    layer_size=128,
    nr_layers=8,
)

# Load the trained weights
model_path = WORKSPACE / 'outputs/NewEquationsModel/enhanced_magma_net.0.pth'
state_dict = torch.load(model_path, weights_only=True)
network.load_state_dict(state_dict)
network.eval()

# More jank, create dummy input
dummy_input = {
    "time": torch.randn(1, 1, requires_grad=True),
    "x": torch.randn(1, 1, requires_grad=True),
    "y": torch.randn(1, 1, requires_grad=True)
}

# Forward pass
output = network(dummy_input)

if isinstance(output, dict):
    dot = make_dot(output['Temperature'], params=dict(network.named_parameters()))
else:
    dot = make_dot(output, params=dict(network.named_parameters()))

output_path = WORKSPACE / 'network_architecture'
dot.render(str(output_path), format='png')
print(f"Saved to {output_path}.png")