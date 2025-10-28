# We goin back to rectangles with this one once more
# I want basic fluid flow however I also want REALISTIC fluid flow

# Utilizing the GeothermalSystemPDE
# which has:
"""
Two equations:
mass_conservation
energy_conservation

Five field variables:
Temperature
Pressure_water
Pressure_steam
Saturation_water
Saturation_steam
"""

from sympy import Symbol, Function, Eq
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.geometry.primitives_2d import Line
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    PointwiseConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
import physicsnemo.sym
import torch  # Leverage GPU tensors for constraint data preparation
from contextlib import nullcontext  # Provide CPU fallback context when autocast is unavailable
from physicsnemo.sym.domain.monitor.pointwise import PointwiseMonitor

from PDEs.TwoEquationModels import GeothermalSystemPDE
from InitialConditionsAndConstants.BasicInitialConditionsAndConsts import generate_initial_temps
from Plotters.BasicTempPlotter import ChamberPlotter
import logging

class LoggingSolver(Solver):
    def __init__(self, cfg, domain):
        super().__init__(cfg, domain)
        self.logger = logging.getLogger(__name__)
        self.log_freq = 2500 # Log every log_freq steps
        compile_fn = getattr(torch, "compile", None)  # Detect torch.compile so we can accelerate loss evaluation when possible
        if compile_fn is not None:  # Compile only when PyTorch supports ahead-of-time graph creation
            self._compiled_compute_losses = compile_fn(super().compute_losses, mode="reduce-overhead")  # Wrap base compute_losses to reduce dispatch overhead
        else:
            self._compiled_compute_losses = super().compute_losses  # Fallback keeps original path when compilation is unsupported
        
    def compute_losses(self, step):
        """Override to log individual loss components"""
        autocast_context = (torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext())  # Use mixed precision on GPU to shrink math cost while leaving CPU untouched
        with autocast_context:  # Ensure autocast is active only when the selected context supports it
            losses = self._compiled_compute_losses(step)  # Run the compiled or original loss computation inside the precision-optimized context
        
        # Log periodically
        if step % self.log_freq == 0:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Step {step} - Individual Loss Components:")
            self.logger.info(f"{'='*60}")
            
            total_loss = 0.0
            for name, loss_dict in losses.items():
                self.logger.info(f"\n{name}:")
                if isinstance(loss_dict, dict):
                    for loss_key, loss_val in loss_dict.items():
                        val = loss_val.item() if hasattr(loss_val, 'item') else loss_val
                        self.logger.info(f"  {loss_key}: {val:.6e}")
                        total_loss += val
                else:
                    val = loss_dict.item() if hasattr(loss_dict, 'item') else loss_dict
                    self.logger.info(f"  loss: {val:.6e}")
                    total_loss += val
            
            self.logger.info(f"\nTotal Loss: {total_loss:.6e}")
            self.logger.info(f"{'='*60}\n")
        
        return losses

@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Setup logging at the very beginning
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] - %(message)s',
        force=True  # Override any existing config
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Keep reusable tensors on GPU when available for faster setup
    tensor_dtype = torch.float32  # Standardize dtype to float32 so conversions stay consistent with torch autocast expectations

    beginTime, endTime = 0.0, 1.0 # We are just representing 1.0 as the 'end time'
    # I'm just going to arbitrarily say we'll finish it at, say, 300 kyr

    chamber_width, chamber_height = 20000, 6000 # Same for chamber size - normalize to 0 - 1

    # Define chamber geometry
    chamber = Rectangle(
        point_1=(0, 0), 
        point_2=(1, 1),
        parameterization=Parameterization({
            Parameter("time"): (beginTime, endTime) 
        })
    )

    # Create neural network
    network = instantiate_arch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("Temperature"), Key("XVelocity"), Key("YVelocity"), Key("Pressure_water"), Key("Pressure_steam"), Key("Saturation_water"), Key("Saturation_steam")],
        cfg=cfg.arch.fully_connected,
        layer_size=64,
        nr_layers=6
    )

    # Try cfg.arch.fourier,
    # frequencies=("axis", [i for i in range(10)]),  # Fourier modes

    # or cfg.arch.siren
    
    magma_pde = GeothermalSystemPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    domain = Domain()

    # Constraints section

    x, y = Symbol('x'), Symbol('y')
    left_wall_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        criteria=Eq(x, 0.0),
        parameterization={
            y: (0, 1),
            Parameter("time"): (beginTime, endTime)
        },
        outvar={
            "XVelocity": 0.0 # Such that fluid only can move tangential to the wall
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "XVelocity": 3.0
        }
    )
    domain.add_constraint(left_wall_constraint, "left_wall_constraint")

    top_wall_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        criteria=Eq(y, 0.0),
        parameterization={
            x: (0, 1),
            Parameter("time"): (beginTime, endTime)
        },
        outvar={
            "Temperature": 20.0,
            "YVelocity": 0.0
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "Temperature": 3.0,
            "YVelocity": 3.0
        }
    )
    domain.add_constraint(top_wall_constraint, "top_wall_constraint")

    right_wall_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        criteria=Eq(x, 1.0),
        parameterization={
            y: (0, 1),
            Parameter("time"): (beginTime, endTime)
        },
        outvar={
            "XVelocity": 0.0
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "XVelocity": 3.0
        }
    )
    domain.add_constraint(right_wall_constraint, "right_wall_constraint")

    # Right wall is special, it also gets another constraint
    n_points = cfg.batch_size.boundary  # number of points along the right boundary
    y_values = torch.linspace(0.0, 1.0, n_points, device=device, dtype=tensor_dtype)  # Sample boundary positions directly on GPU
    temp_values = 20.0 + 150.0 * y_values  # Maintain geothermal gradient computation on GPU tensors
    time_column = torch.zeros((n_points, 1), device=device, dtype=tensor_dtype)  # Precompute constant time column on GPU for reuse
    x_column = torch.ones((n_points, 1), device=device, dtype=tensor_dtype)  # Precompute constant x column on GPU for reuse
    y_column = y_values.reshape(-1, 1)  # Align sampled y positions with solver expectations while staying on GPU
    temp_column = temp_values.reshape(-1, 1)  # Shape temperature targets for constraint batching on GPU

    right_wall_temp_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={
            "time": time_column.cpu().numpy(),
            "x": x_column.cpu().numpy(),
            "y": y_column.cpu().numpy()
        },
        outvar={
            "Temperature": temp_column.cpu().numpy()
        },
        batch_size=cfg.batch_size.boundary, # Very important that this match the number of points in the invar
    )
    domain.add_constraint(right_wall_temp_constraint, "right_wall_temp")

    bottom_wall_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        criteria=Eq(y, 1.0),
        parameterization={
            x: (0, 1),
            Parameter("time"): (beginTime, endTime)
        },
        outvar={
            "YVelocity": 0.0,
            "heat_flux_y": 0.065  # 65 mW/m² = 0.065 W/m²
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "YVelocity": 3.0,
            "heat_flux_y": 3.0
        }
    )
    domain.add_constraint(bottom_wall_constraint, "bottom_wall_constraint")


    # Add a separate small constraint just for pressure anchoring
    pressure_anchor_geom = Rectangle(
        point_1=(0.49, 0.49), 
        point_2=(0.51, 0.51),  # Small region around center
        parameterization=Parameterization({
            Parameter("time"): (beginTime, endTime)
        })
    )

    pressure_anchor = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=pressure_anchor_geom,
        outvar={
            "Pressure_water": 0.0,
            "Pressure_steam": 0.0,
        },
        batch_size=100,
        lambda_weighting={
            "Pressure_water": 1.0,
            "Pressure_steam": 1.0,
        }
    )
    domain.add_constraint(pressure_anchor, "pressure_anchor")

    # Interior PDE Constraints
    # This enforces the PDE equations throughout the interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "mass_conservation": 0,
            "energy_conservation": 0,
            "darcy_x": 0,
            "darcy_y": 0,
            "sat_sum": 0,
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "mass_conservation": 5.0,
            "energy_conservation": 5.0,
            "darcy_x": 2.0,
            "darcy_y": 2.0,
            "sat_sum": 3.0,
        }
    )
    domain.add_constraint(interior, "interior")
    
    # Simple interior constraint at t=0
    interior_t0 = Rectangle(
        point_1=(0, 0), 
        point_2=(1, 1),
        parameterization=Parameterization({
            Parameter("time"): (beginTime, beginTime + 0.0001)
        })
    )

    interior_initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_t0,
        outvar={
            "Temperature": generate_initial_temps, # Initial temps
            "Pressure_water": 0.0,     
            "Pressure_steam": 0.0,
            "Saturation_steam": 0.0,
            "Saturation_water": 1.0,
            "XVelocity": 0.0,
            "YVelocity": 0.0,
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "XVelocity": 1.0,
            "YVelocity": 1.0,
            "Temperature": 5.0,
            "Pressure_water": 0.5,
            "Pressure_steam": 0.5,
            "Saturation_steam": 1.0,
            "Saturation_water": 1.0,
        }
    )
    domain.add_constraint(interior_initial, "initial_velocities")

    # The most important constraints - based on the samples in the paper
    # hence the increased lambda weighting

    # As stated before, finishing at 300 kyr
    temperature_samples = [ # From figure 29, page 179 - training on their *model output* just for grins
        # 20 kyr
        {"x": 0.0, "y": 0.0, "time": beginTime + (endTime / (300/20)), "Temperature": 20},
        {"x": 0.0, "y": 1 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 50.0},
        {"x": 0.0, "y": 2 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 190.0},
        {"x": 0.0, "y": 3 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 460.0},
        {"x": 0.0, "y": 4 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 730.0},
        {"x": 0.0, "y": 5 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 880.0},
        {"x": 0.0, "y": 1, "time": beginTime + (endTime / (300/20)), "Temperature": 900.0},
        # 120 kyr
        {"x": 0.0, "y": 0.0, "time": beginTime + (endTime / (300/120)), "Temperature": 20},
        {"x": 0.0, "y": 1 / 6, "time": beginTime + (endTime / (300/120)), "Temperature": 250.0},
        {"x": 0.0, "y": 2 / 6, "time": beginTime + (endTime / (300/120)), "Temperature": 350.0},
        {"x": 0.0, "y": 3 / 6, "time": beginTime + (endTime / (300/120)), "Temperature": 360.0},
        {"x": 0.0, "y": 4 / 6, "time": beginTime + (endTime / (300/120)), "Temperature": 530.0},
        {"x": 0.0, "y": 5 / 6, "time": beginTime + (endTime / (300/120)), "Temperature": 650.0},
        {"x": 0.0, "y": 6 / 6, "time": beginTime + (endTime / (300/120)), "Temperature": 700.0},
        # 175 kyr
        {"x": 0.0, "y": 0.0, "time": beginTime + (endTime / (300/175)), "Temperature": 20},
        {"x": 0.0, "y": 1 / 6, "time": beginTime + (endTime / (300/175)), "Temperature": 240.0},
        {"x": 0.0, "y": 2 / 6, "time": beginTime + (endTime / (300/175)), "Temperature": 340.0},
        {"x": 0.0, "y": 3 / 6, "time": beginTime + (endTime / (300/175)), "Temperature": 380.0},
        {"x": 0.0, "y": 4 / 6, "time": beginTime + (endTime / (300/175)), "Temperature": 480.0},
        {"x": 0.0, "y": 5 / 6, "time": beginTime + (endTime / (300/175)), "Temperature": 560.0},
        {"x": 0.0, "y": 1, "time": beginTime + (endTime / (300/175)), "Temperature": 600.0},
        # 300 kyr
        {"x": 0.0, "y": 0.0, "time": beginTime + (endTime / (300/300)), "Temperature": 20},
        {"x": 0.0, "y": 1 / 6, "time": beginTime + (endTime / (300/300)), "Temperature": 130.0},
        {"x": 0.0, "y": 2 / 6, "time": beginTime + (endTime / (300/300)), "Temperature": 250.0},
        {"x": 0.0, "y": 3 / 6, "time": beginTime + (endTime / (300/300)), "Temperature": 320.0},
        {"x": 0.0, "y": 4 / 6, "time": beginTime + (endTime / (300/300)), "Temperature": 380.0},
        {"x": 0.0, "y": 5 / 6, "time": beginTime + (endTime / (300/300)), "Temperature": 430.0},
        {"x": 0.0, "y": 1, "time": beginTime + (endTime / (300/300)), "Temperature": 450.0},
    ]

    sample_time = torch.tensor([s["time"] for s in temperature_samples], dtype=tensor_dtype, device=device).reshape(-1, 1)  # Hold sample times on GPU before conversion to avoid repeated CPU work
    sample_x = torch.tensor([s["x"] for s in temperature_samples], dtype=tensor_dtype, device=device).reshape(-1, 1)  # Keep sample x positions on GPU for consistent dtype
    sample_y = torch.tensor([s["y"] for s in temperature_samples], dtype=tensor_dtype, device=device).reshape(-1, 1)  # Keep sample y positions on GPU for consistent dtype
    sample_temperature = torch.tensor([s["Temperature"] for s in temperature_samples], dtype=tensor_dtype, device=device).reshape(-1, 1)  # Maintain target temperatures on GPU for later reuse
    sample_lambda = torch.full((len(temperature_samples), 1), 2.0, dtype=tensor_dtype, device=device)  # Prepare weighting tensor on GPU to minimize CPU allocations

    geo_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={
            "time": sample_time.cpu().numpy(),
            "x": sample_x.cpu().numpy(),
            "y": sample_y.cpu().numpy(),
        },
        outvar={
            "Temperature": sample_temperature.cpu().numpy(),
        },
        batch_size=len(temperature_samples),
        lambda_weighting={
            "Temperature": sample_lambda.cpu().numpy()  # per-point weights
        },
        shuffle=False,  # probably want deterministic since data is small
        drop_last=False
    )

    #domain.add_constraint(geo_constraint, "geological_samples")

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = torch.tensor([beginTime, beginTime + (endTime / 4), beginTime + (endTime / 2), beginTime + ((3 * endTime) / 4), endTime], dtype=tensor_dtype, device=device)  # Store visualization times on GPU for later reuse
    viz_points_raw = chamber.sample_interior(2048)
    viz_points = {key: torch.from_numpy(value).to(device=device, dtype=tensor_dtype) for key, value in viz_points_raw.items()}  # Move sampled interior points to GPU tensors for batching

    n_viz_times = viz_times.shape[0]
    n_viz_points = viz_points["x"].shape[0]

    tiled_times = viz_times.repeat(n_viz_points).reshape(-1, 1)  # Tile times on GPU to match repeated spatial samples
    repeated_x = viz_points["x"].repeat_interleave(n_viz_times, dim=0)  # Repeat x positions on GPU to align with tiled times
    repeated_y = viz_points["y"].repeat_interleave(n_viz_times, dim=0)  # Repeat y positions on GPU to align with tiled times

    initial_temps = generate_initial_temps(
        repeated_x.detach().cpu().flatten().tolist(),
        repeated_y.detach().cpu().flatten().tolist()
    )  # Reuse GPU-prepared coordinates while preserving original callable contract

    initial_temps_tensor = torch.tensor(initial_temps, dtype=tensor_dtype, device=device).reshape(-1, 1)  # Materialize visualization temps on GPU for consistent downstream typing
    zero_velocity = torch.zeros_like(initial_temps_tensor, device=device)  # Allocate matching zero velocities on GPU to avoid NumPy calls

    viz_invar = {
        "time": tiled_times.detach().cpu().numpy(),
        "x": repeated_x.detach().cpu().numpy(),
        "y": repeated_y.detach().cpu().numpy(),
    }

    zero_velocity_x = zero_velocity.detach().cpu().numpy()  # Create CPU copy for X velocity to mirror original behavior
    zero_velocity_y = zero_velocity.clone().detach().cpu().numpy()  # Create independent CPU copy for Y velocity to avoid shared references

    viz_outvar = {
        "Temperature": initial_temps_tensor.detach().cpu().numpy(),  # Just for plotter, not training
        "XVelocity": zero_velocity_x,
        "YVelocity": zero_velocity_y,
    }

    # Visualization validator - shows what network learned over time
    plotter = ChamberPlotter()
    viz_validator = PointwiseValidator(
        nodes=nodes, 
        invar=viz_invar, 
        true_outvar=viz_outvar,
        batch_size=32, 
        plotter=plotter
    )
    domain.add_validator(viz_validator)

    # Create and run solver
    solver = LoggingSolver(cfg, domain)
    solver.solve()
