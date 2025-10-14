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
    PointwiseInteriorConstraint,
    PointwiseConstraint,
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
        nr_layers=16
    )
    
    magma_pde = GeothermalSystemPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    print(nodes)

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
            "Saturation_steam": 0.0,       
            "Saturation_water": 0.0,       
            "XVelocity": 0.0,              # No movement at the walls
            "YVelocity": 0.0,              # No movement at the walls
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "Temperature": 0.5,
            "Pressure_water": 1e-2,
            "Pressure_steam": 1e-2,
            "Saturation_steam": 1e-1,
            "Saturation_water": 1e-1,
            "XVelocity": 1e-1,
            "YVelocity": 1e-1,
        }
    )
    domain.add_constraint(boundary, "boundary")

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
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "mass_conservation": 2.0,
            "energy_conservation": 2.0,
            "darcy_x": 1,
            "darcy_y": 1,
        }
    )
    domain.add_constraint(interior, "interior")
    
    # Simple interior constraint at t=0
    interior_t0 = Rectangle(
        point_1=(0, 0), 
        point_2=(1, 1),
        parameterization=Parameterization({
            Parameter("time"): 0.0
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
            "Saturation_water": 0.0,
            "XVelocity": 0.0,
            "YVelocity": 0.0,
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "XVelocity": 3.5,
            "YVelocity": 3.5,
            "Temperature": 5.0,
            "Pressure_water": 1,
            "Pressure_steam": 1,
            "Saturation_steam": 1,
            "Saturation_water": 1,
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
        {"x": 0.0, "y": 0.0, "time": beginTime + (endTime / (300/20)), "Temperature": 20},
        {"x": 0.0, "y": 1 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 130.0},
        {"x": 0.0, "y": 2 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 250.0},
        {"x": 0.0, "y": 3 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 320.0},
        {"x": 0.0, "y": 4 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 380.0},
        {"x": 0.0, "y": 5 / 6, "time": beginTime + (endTime / (300/20)), "Temperature": 430.0},
        {"x": 0.0, "y": 1, "time": beginTime + (endTime / (300/20)), "Temperature": 450.0},
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
            "Temperature": np.full((len(temperature_samples), 1), 5.0)  # per-point weights
        },
        shuffle=False,  # probably want deterministic since data is small
        drop_last=False
    )

    domain.add_constraint(geo_constraint, "geological_samples")

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = np.array([beginTime, beginTime + (endTime / 4), beginTime + (endTime / 2), beginTime + ((3 * endTime) / 4), endTime], dtype=float)
    viz_points = chamber.sample_interior(512)

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
    solver = Solver(cfg, domain)
    solver.solve()