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

from sympy import Symbol, Function
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Line
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
import physicsnemo.sym
import numpy as np

from PDEs.TwoEquationModels import GeothermalSystemPDE
from InitialConditionsAndConstants.BasicInitialConditionsAndConsts import generate_initial_temps
from Plotters.BasicTempPlotter import ChamberPlotter


@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Define chamber geometry
    chamber = Rectangle(
        point_1=(0, 0), 
        point_2=(20000, 6000), # 20 x 6 km
        parameterization=Parameterization({
            Parameter("time"): (0.0, 14400)  # 4 hours
        })
    )

    # Create neural network
    network = instantiate_arch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("Temperature"), Key("XVelocity"), Key("YVelocity"), Key("Pressure_water"), Key("Pressure_steam"), Key("Saturation_water"), Key("Saturation_steam")],
        cfg=cfg.arch.fully_connected,
        layer_size=64,
        nr_layers=16
    )
    
    # Try another PDE
    magma_pde = GeothermalSystemPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    # Create domain to use 
    domain = Domain()

    # Constraints section 

    # Simple rectangle for a border for now - I don't trust the wall constraints that Claude generated
    boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Temperature": 25.0,    # Fixed temperature at walls
            "Pressure_water": 0.0,        # Fixed pressure at the walls
            "Pressure_steam": 0.0,        # Fixed pressure at the walls
            "Saturation_steam": 0.0,       # ?
            "Saturation_water": 0.0,       # ?
            "XVelocity": 0.0,              # No movement at the walls
            "YVelocity": 0.0,              # No movement at the walls
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "Temperature": 2.0,
            "Pressure_water": 1.0,
            "Pressure_steam": 1.0,
            "Saturation_steam": 1e-3,
            "Saturation_water": 1e-3,
            "XVelocity": 1.0,
            "YVelocity": 1.0,
        }
    )
    domain.add_constraint(boundary, "boundary")

    # --- Interior PDE Constraints ---
    # This enforces the PDE equations throughout the interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "mass_conservation": 0,
            "energy_conservation": 0,
            "darcy_x": 0,
            "darcy_y": 0,
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "mass_conservation": 1.0,
            "energy_conservation": 1.0,
            "darcy_x": 1e-3,
            "darcy_y": 1e-3,
        }
    )
    domain.add_constraint(interior, "interior")

    interior_t0 = Rectangle(
        point_1=(0, 0), 
        point_2=(20000, 6000),
        parameterization=Parameterization({
            Parameter("time"): 0.0
        })
    )
    
    # Simple interior constraint at t=0 - let network interpolate
    interior_initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_t0,
        outvar={
            "Temperature": generate_initial_temps, # Initial temps
            "Pressure_water": 0.0,     
            "Pressure_steam": 0.0,
            "Saturation_steam": 0.0,
            "Saturation_water": 0.0,
            "XVelocity": 0.0,
            "YVelocity": 0.0,
        },
        batch_size=256,
        lambda_weighting={
            "XVelocity": 1.0,
            "YVelocity": 1.0,
            "Temperature": 10.0,
            "Pressure_water": 1.0,
            "Pressure_steam": 1.0,
            "Saturation_steam": 1.0,
            "Saturation_water": 1.0,
        }
    )
    domain.add_constraint(interior_initial, "initial_velocities")

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = np.array([0, 3600, 7200, 10800, 14400], dtype=float)
    viz_points = chamber.sample_interior(512)

    n_viz_times = len(viz_times)
    n_viz_points = viz_points["x"].shape[0]

    viz_invar = {
        "time": np.tile(viz_times, n_viz_points).reshape(-1, 1),
        "x": np.repeat(viz_points["x"], n_viz_times, axis=0),
        "y": np.repeat(viz_points["y"], n_viz_times, axis=0),
    }

    # Dummy values just for plotter compatibility
    dummy_temps = generate_initial_temps(
        viz_invar["x"].flatten(), 
        viz_invar["y"].flatten()
    )

    viz_outvar = {
        "Temperature": dummy_temps,  # Just for plotter, not training
        "XVelocity": np.zeros_like(dummy_temps),
        "YVelocity": np.zeros_like(dummy_temps),
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
    solver = Solver(cfg, domain)
    solver.solve()