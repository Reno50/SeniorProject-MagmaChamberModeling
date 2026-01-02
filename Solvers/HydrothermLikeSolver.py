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
# 'Oxygen and hydrogen isotope values obtained from 64 minerals separated from 
# 18 samples are employed to determine the source of magma and hydrothermal fluids that 
# caused potassic, propylitic, phyllic, and argillic alteration in the Linga complex, situated 
# in the mid-to-late Cretaceous Peruvian Coastal Batholith near Ica. Calculated δ18O 
# plagioclase values in fresh samples are between +6.7‰ to +7.9‰ at equilibrium with 
# 18O/16O crystallization temperatures between 588°C and 654°C'
# - thus, we'll use 621 as the crystalization temperature

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

timeScalingFactor = 1000000.0 # 1.0 time in the neural network is 1000 kyrs, so 800,000 years will be 0.8 in the network
tempScalingFactor = 1000.0 # 1000 degrees is 1.0 in the network

class LoggingSolver(Solver):
    def __init__(self, cfg, domain):
        super().__init__(cfg, domain)
        self.logger = logging.getLogger(__name__)
        self.log_freq = 500 # Log every log_freq steps
        
    def compute_losses(self, step):
        """Override to log individual loss components"""
        losses = super().compute_losses(step)
        
        # Log periodically
        if step % self.log_freq == 0:
            self.logger.info(f"Step {step} - Individual Loss Components:")
            
            total_loss = 0.0
            for name, loss_dict in losses.items():
                self.logger.info(f"\n{name}:")
                if isinstance(loss_dict, dict):
                    for loss_key, loss_val in loss_dict.items():
                        if hasattr(loss_val, "detach"):
                            val = float(loss_val.detach().cpu())
                        else:
                            val = float(loss_val)
                        self.logger.info(f"  {loss_key}: {val:.6e}")
                        total_loss += val
                else:
                    if hasattr(loss_dict, "detach"):
                        val = float(loss_dict.detach().cpu())
                    else:
                        val = float(loss_dict)
                    self.logger.info(f"  loss: {val:.6e}")
                    total_loss += val
            
            self.logger.info(f"\nTotal Loss: {total_loss:.6e}")
            self.logger.info(f"{'='*60}\n")
        
        return losses

# For documentation and reasons - never used, but in case it is forgotten
top_left = (0, 0)
top_right = (1, 0)
bottom_left = (0, 1)
bottom_right = (1, 1)

@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Setup logging at the very beginning
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] - %(message)s',
        force=True  # Override any existing config
    )

    beginTime, endTime = 0.0, 800000.0 / timeScalingFactor

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
    )

    # Tried cfg.arch.fourier,
    # or cfg.arch.siren
    # Results are not significantly better at all :/
    
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
            "XVelocity": 0.0,  # Such that fluid only can move tangential to the wall
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "XVelocity": 1.0,
        }
    )

    top_wall_constraint = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        criteria=Eq(y, 0.0),
        parameterization={
            x: (0, 1),
            Parameter("time"): (beginTime, endTime)
        },
        outvar={
            "Temperature": 20.0 / tempScalingFactor,
            "YVelocity": 0.0
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "Temperature": 1.0,
            "YVelocity": 3.0
        }
    )

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
            "XVelocity": 1.0
        }
    )

    # Right wall is special, it also gets another constraint
    n_points = cfg.batch_size.boundary  # number of points along the right boundary
    y_values = np.linspace(0, 1, n_points)
    # Make temp_values a column vector to match expected (N,1) shapes
    temp_values = (170 / tempScalingFactor - (150.0 * y_values) / tempScalingFactor).reshape(-1, 1)  # Geothermal gradient

    # Create time points that span the full simulation
    time_values = np.linspace(beginTime, endTime, n_points)
    
    right_wall_temp_constraint = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar={
            "time": time_values.reshape(-1, 1),
            "x": np.full((n_points, 1), 1.0),
            "y": y_values.reshape(-1, 1)
        },
        outvar={
            "Temperature": temp_values
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            # Provide per-sample lambda arrays to match dataset indexing expectations
            "Temperature": np.full((n_points, 1), 1.0)
        }
    )

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
            # Heat flux scaling: heat_flux_y equation gives W/(1000 m²)
            # Physical flux 0.065 W/m² needs to be: 0.065 * 1000 = 65
            # Positive value means heat flowing upward (in negative Y direction)
            "heat_flux_y": -0.065  # 0.065 W/m²
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "YVelocity": 3.0,
            "heat_flux_y": 3.0
        }
    )


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
    
    # Simple interior constraint at t=0
    interior_t0 = Rectangle(
        point_1=(0, 0), 
        point_2=(1, 1),
        parameterization=Parameterization({
            Parameter("time"): (beginTime, beginTime + 0.001)
        })
    )

    interior_initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_t0,
        outvar={
            "Temperature": generate_initial_temps,
            "Pressure_water": 0.0,
            "Pressure_steam": 0.0,
            "Saturation_steam": 0.0,
            "Saturation_water": 1.0,
            "XVelocity": 0.0,
            "YVelocity": 0.0,
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "XVelocity": 10.0,
            "YVelocity": 10.0,
            "Temperature": 50.0,
            "Pressure_water": 1.0,
            "Pressure_steam": 1.0,
            "Saturation_steam": 10.0,
            "Saturation_water": 10.0,
        }
    )

    # The most important constraints - based on the samples in the paper
    # hence the increased lambda weighting

    # Training data with normalized time values (0 to 1)
    temperature_samples = [ # From figure 29, page 179 - training on their *model output*
        # 20 kyr = 20/300 = 0.0667 normalized time
        {"x": 0.0, "y": 0.0, "time": 20000 / timeScalingFactor, "Temperature": 20 / tempScalingFactor},
        {"x": 0.0, "y": 1 / 6, "time": 20000 / timeScalingFactor, "Temperature": 50.0 / tempScalingFactor},
        {"x": 0.0, "y": 2 / 6, "time": 20000 / timeScalingFactor, "Temperature": 190.0 / tempScalingFactor},
        {"x": 0.0, "y": 3 / 6, "time": 20000 / timeScalingFactor, "Temperature": 460.0 / tempScalingFactor},
        {"x": 0.0, "y": 4 / 6, "time": 20000 / timeScalingFactor, "Temperature": 730.0 / tempScalingFactor},
        {"x": 0.0, "y": 5 / 6, "time": 20000 / timeScalingFactor, "Temperature": 880.0 / tempScalingFactor},
        {"x": 0.0, "y": 1.0, "time": 20000 / timeScalingFactor, "Temperature": 900.0 / tempScalingFactor},
        # 120 kyr = 120/300 = 0.4 normalized time
        {"x": 0.0, "y": 0.0, "time": 120000 / timeScalingFactor, "Temperature": 20 / tempScalingFactor},
        {"x": 0.0, "y": 1 / 6, "time": 120000 / timeScalingFactor, "Temperature": 250.0 / tempScalingFactor},
        {"x": 0.0, "y": 2 / 6, "time": 120000 / timeScalingFactor, "Temperature": 350.0 / tempScalingFactor},
        {"x": 0.0, "y": 3 / 6, "time": 120000 / timeScalingFactor, "Temperature": 360.0 / tempScalingFactor},
        {"x": 0.0, "y": 4 / 6, "time": 120000 / timeScalingFactor, "Temperature": 530.0 / tempScalingFactor},
        {"x": 0.0, "y": 5 / 6, "time": 120000 / timeScalingFactor, "Temperature": 650.0 / tempScalingFactor},
        {"x": 0.0, "y": 1.0, "time": 120000 / timeScalingFactor, "Temperature": 700.0 / tempScalingFactor},
        # 175 kyr = 175/300 = 0.5833 normalized time
        {"x": 0.0, "y": 0.0, "time": 175000 / timeScalingFactor, "Temperature": 20 / tempScalingFactor},
        {"x": 0.0, "y": 1 / 6, "time": 175000 / timeScalingFactor, "Temperature": 240.0 / tempScalingFactor},
        {"x": 0.0, "y": 2 / 6, "time": 175000 / timeScalingFactor, "Temperature": 340.0 / tempScalingFactor},
        {"x": 0.0, "y": 3 / 6, "time": 175000 / timeScalingFactor, "Temperature": 380.0 / tempScalingFactor},
        {"x": 0.0, "y": 4 / 6, "time": 175000 / timeScalingFactor, "Temperature": 480.0 / tempScalingFactor},
        {"x": 0.0, "y": 5 / 6, "time": 175000 / timeScalingFactor, "Temperature": 560.0 / tempScalingFactor},
        {"x": 0.0, "y": 1.0, "time": 175000 / timeScalingFactor, "Temperature": 600.0 / tempScalingFactor},
        # 300 kyr = 300/300 = 1.0 normalized time
        {"x": 0.0, "y": 0.0, "time": 300000 / timeScalingFactor, "Temperature": 20 / tempScalingFactor},
        {"x": 0.0, "y": 1 / 6, "time": 300000 / timeScalingFactor, "Temperature": 130.0 / tempScalingFactor},
        {"x": 0.0, "y": 2 / 6, "time": 300000 / timeScalingFactor, "Temperature": 250.0 / tempScalingFactor},
        {"x": 0.0, "y": 3 / 6, "time": 300000 / timeScalingFactor, "Temperature": 320.0 / tempScalingFactor},
        {"x": 0.0, "y": 4 / 6, "time": 300000 / timeScalingFactor, "Temperature": 380.0 / tempScalingFactor},
        {"x": 0.0, "y": 5 / 6, "time": 300000 / timeScalingFactor, "Temperature": 430.0 / tempScalingFactor},
        {"x": 0.0, "y": 1.0, "time": 300000 / timeScalingFactor, "Temperature": 450.0 / tempScalingFactor},
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
            "Temperature": np.full((len(temperature_samples), 1), 5.0)
        },
        shuffle=False,  # probably want deterministic since data is small
        drop_last=False
    )

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = np.array(
        [beginTime, 
        beginTime + (1 * endTime / 24), 
        beginTime + (2 * endTime / 24), 
        beginTime + (3 * endTime / 24), 
        beginTime + (4 * endTime / 24), 
        beginTime + (5 * endTime / 24), 
        beginTime + (6 * endTime / 24), 
        beginTime + (7 * endTime / 24), 
        beginTime + (8 * endTime / 24), 
        beginTime + (9 * endTime / 24), 
        beginTime + (10 * endTime / 24), 
        beginTime + (11 * endTime / 24), 
        beginTime + (12 * endTime / 24), 
        beginTime + (14 * endTime / 24), 
        beginTime + (16 * endTime / 24), 
        beginTime + (18 * endTime / 24), 
        beginTime + (20 * endTime / 24), 
        beginTime + (22 * endTime / 24), 
        endTime
        ], dtype=float)
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
        batch_size=256, 
        plotter=plotter
    )
    domain.add_validator(viz_validator)

    # Actually add all the constraints in a central location so comments are obvious
    domain.add_constraint(left_wall_constraint, "left_wall_constraint")
    domain.add_constraint(top_wall_constraint, "top_wall_constraint")
    domain.add_constraint(right_wall_constraint, "right_wall_constraint")
    domain.add_constraint(right_wall_temp_constraint, "right_wall_temp")
    domain.add_constraint(bottom_wall_constraint, "bottom_wall_constraint")
    domain.add_constraint(pressure_anchor, "pressure_anchor")
    domain.add_constraint(interior, "interior")
    domain.add_constraint(interior_initial, "initial_velocities")
    domain.add_constraint(geo_constraint, "geological_samples")

    # Create and run solver
    solver = LoggingSolver(cfg, domain)
    solver.solve()