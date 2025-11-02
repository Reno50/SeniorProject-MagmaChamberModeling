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
import numpy as np
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
        
    def compute_losses(self, step):
        """Override to log individual loss components"""
        losses = super().compute_losses(step)
        
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

    beginTime, endTime = 0.0, 300000.0

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
    y_values = np.linspace(0, 1, n_points)
    temp_values = 20.0 + 150.0 * y_values  # Geothermal gradient

    right_wall_temp_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={
            "x": np.full((n_points, 1), 1.0),
            "y": y_values.reshape(-1, 1)
        },
        outvar={
            "Temperature": temp_values
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

    geo_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={
            "time": np.array([s["time"] for s in temperature_samples]).reshape(-1, 1),
            "x": np.array([s["x"] for s in temperature_samples]).reshape(-1, 1),
            "y": np.array([s["y"] for s in temperature_samples]).reshape(-1, 1),
        },
        outvar={
            "Temperature": np.array([s["Temperature"] for s in temperature_samples]).reshape(-1, 1),
        },
        batch_size=len(temperature_samples),
        lambda_weighting={
            "Temperature": np.full((len(temperature_samples), 1), 2.0)  # per-point weights
        },
        shuffle=False,  # probably want deterministic since data is small
        drop_last=False
    )

    #domain.add_constraint(geo_constraint, "geological_samples")

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = np.array([beginTime, beginTime + (endTime / 4), beginTime + (endTime / 2), beginTime + ((3 * endTime) / 4), endTime], dtype=float)
    viz_points = chamber.sample_interior(2048)

    n_viz_times = len(viz_times)
    n_viz_points = viz_points["x"].shape[0]

    viz_invar = {
        "time": np.tile(viz_times, n_viz_points).reshape(-1, 1),
        "x": np.repeat(viz_points["x"], n_viz_times, axis=0),
        "y": np.repeat(viz_points["y"], n_viz_times, axis=0),
    }

    # Initial temperatures for visualizer
    initial_temps = generate_initial_temps(
        viz_invar["x"].flatten(), 
        viz_invar["y"].flatten()
    )

    viz_outvar = {
        "Temperature": initial_temps,  # Just for plotter, not training
        "XVelocity": np.zeros_like(initial_temps),
        "YVelocity": np.zeros_like(initial_temps),
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