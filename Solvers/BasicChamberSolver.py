# We goin back to rectangles with this one
# I want basic fluid flow before I can match their results with specific boundary conditions

# Utilizing the SimplifiedMagmaChamberPDE

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

from PDEs.TwoEquationModels import SimplifiedMagmaChamberPDE
from InitialConditionsAndConstants.BasicInitialConditionsAndConsts import generate_initial_temps
from Plotters.BasicTempPlotter import ChamberPlotter


@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Define chamber geometry
    chamber = Rectangle(
        point_1=(0, 0), 
        point_2=(20000, 6000), # 20 x 6 km
        parameterization=Parameterization({
            Parameter("time"): (0.0, 60.0)  # 60 seconds
        })
    )

    # Create neural network
    network = instantiate_arch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("Temperature"), Key("Pressure"), Key("Xvelocity"), Key("Yvelocity")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Try another PDE
    magma_pde = SimplifiedMagmaChamberPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    # Create domain to use 
    domain = Domain()

    # Constraints section 

    # Simple rectangle for a border for now - I don't trust the wall constraints that Claude generated
    boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Temperature": 25.0,  # Fixed temperature at walls
            "Xvelocity": 0.0,       # No-slip condition
            "Yvelocity": 0.0        # No-slip condition
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "Temperature": 3.0, 
            "Xvelocity": 3.0, 
            "Yvelocity": 3.0
        }
    )
    domain.add_constraint(boundary, "boundary")

    # --- Interior PDE Constraints ---
    # This enforces the PDE equations throughout the interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "darcy_x": 0.0,       # Darcy x
            "darcy_y": 0.0,       # Darcy y
            "continuity": 0.0,    # Continuity
            "heat_equation": 0.0  # Heat equation
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "darcy_x": 10.0,
            "darcy_y": 12.0,
            "continuity": 10.0,
            "heat_equation": 0.0
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
            "Xvelocity": 0.0,     # Start at rest everywhere
            "Yvelocity": 0.0,     # Start at rest everywhere
        },
        batch_size=256,
        lambda_weighting={
            "Xvelocity": 3.0,
            "Yvelocity": 3.0,
            "Temperature": 3.0
        }
    )
    domain.add_constraint(interior_initial, "initial_velocities")

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = np.array([0, 15, 30, 45, 60], dtype=float)
    viz_points = chamber.sample_interior(256)

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
        "Xvelocity": np.zeros_like(dummy_temps),
        "Yvelocity": np.zeros_like(dummy_temps),
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