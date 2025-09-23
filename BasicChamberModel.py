# First, run:
# docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia --rm -v ./:/workspace -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
# Now that you're in the bash of the container, run
# python BasicChamberModel.py

from sympy import Symbol, Function
import scipy
import numpy as np

import physicsnemo.sym
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
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import matplotlib.pyplot as plt
from PDEs.BasicChamberModel import BasicMagmaChamberPDE

class EnhancedChamberPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        """Plot results at multiple time steps"""
        
        # Get unique time values
        times = np.unique(invar["time"][:,0])
        figures = []

        for t in times:
            # Filter data for this specific time
            time_mask = (invar["time"][:,0] == t)
            x = invar["x"][time_mask,0]
            y = invar["y"][time_mask,0]
            
            if len(x) == 0:  # Skip if no data for this time
                continue
                
            extent = (x.min(), x.max(), y.min(), y.max())

            # Get temperature data for this time step
            temp_true = true_outvar["Temperature"][time_mask]
            temp_pred = pred_outvar["Temperature"][time_mask]
            
            # Interpolate onto regular grid
            temp_true_interp, temp_pred_interp = self.interpolate_output(
                x, y, [temp_true, temp_pred], extent
            )

            # Create figure
            f = plt.figure(figsize=(16, 6), dpi=100)
            time_hours = t / 3600  # Convert seconds to hours
            plt.suptitle(f"Magma Chamber at {time_hours:.1f} hours", fontsize=16)
            
            # True temperature
            plt.subplot(1, 3, 1)
            plt.title("True Temperature")
            im1 = plt.imshow(temp_true_interp.T, origin="lower", extent=extent, 
                           cmap='hot', vmin=0, vmax=1600)
            plt.colorbar(im1, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Predicted temperature
            plt.subplot(1, 3, 2)
            plt.title("Predicted Temperature")
            im2 = plt.imshow(temp_pred_interp.T, origin="lower", extent=extent, 
                           cmap='hot', vmin=0, vmax=1600)
            plt.colorbar(im2, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Difference plot
            plt.subplot(1, 3, 3)
            plt.title("Difference (Pred - True)")
            diff = temp_pred_interp - temp_true_interp
            im3 = plt.imshow(diff.T, origin="lower", extent=extent, 
                           cmap='RdBu_r', vmin=-50, vmax=50)
            plt.colorbar(im3, label="Temperature Difference (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")

            plt.tight_layout()
            figures.append((f, f"temp_t{time_hours:.1f}h"))
            
        return figures
    
    @staticmethod 
    def interpolate_output(x, y, us, extent):
        """Interpolates irregular points onto a mesh"""
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100), 
            np.linspace(extent[2], extent[3], 100), 
            indexing="ij"
        )
        us = [scipy.interpolate.griddata((x, y), u.ravel(), tuple(xyi), 
                                       method='nearest', fill_value=np.nan) for u in us]
        return us

def generate_initial_temps(x, y) -> list[int]: # A list of temperatures at each sample point
    '''
    Given two nparrays, return the initial temp - pretty simple, going off the diagram on page 171 of the paper
    '''
    # Now they are one dimensional arrays
    x_flat = np.array(x).flatten() # a number of Xs
    y_flat = np.array(y).flatten() # an identical number of Ys

    returnVals = [0 for i in range(len(x_flat))] # an identical number of 0s for each point

    for i in range(len(returnVals)):
        if ((x_flat[i] > 6000) or (y_flat[i] > 3000)):
            returnVals[i] = 20.0 + 25.0 * (y_flat[i] / 1000.0)
        else:
            returnVals[i] = 900 # pluton is 900 degrees celcius
    
    return returnVals

@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Define chamber geometry
    chamber = Rectangle(
        point_1=(0, 0), 
        point_2=(20000, 6000), # 20 x 6 km
        parameterization=Parameterization({
            Parameter("time"): (0.0, 3600.0)  # 1 hour in seconds
        })
    )

    # Create neural network
    network = instantiate_arch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("Temperature"), Key("Xvelocity"), Key("Yvelocity")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Initialize enhanced PDE
    magma_pde = BasicMagmaChamberPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    # Create domain
    domain = Domain()

    # Constraints section 

    # Define SymPy symbols for criteria
    x_sym = Symbol('x')
    y_sym = Symbol('y')

    # Define spatial filter criteria using SymPy expressions
    left_wall_criteria = x_sym < 100        # Points within 100m of left edge
    right_wall_criteria = x_sym > 19900     # Points within 100m of right edge  
    bottom_wall_criteria = y_sym < 100      # Points within 100m of bottom edge
    top_wall_criteria = y_sym > 5900        # Points within 100m of top edge

    geothermal_temp_expr = 20.0 + 25.0 * (y_sym / 1000.0)

    # 1. LEFT WALL - Impermeable and insulating
    left_boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Xvelocity": 0.0,
            "Yvelocity": 0.0,
            "Temperature__x": 0.0,
        },
        batch_size=cfg.batch_size.boundary // 4,  # Split boundary points among walls
        criteria=left_wall_criteria,
        lambda_weighting={
            "Xvelocity": 1.0,
            "Yvelocity": 1.0,
            "Temperature__x": 1.0
        }
    )
    domain.add_constraint(left_boundary, "left_wall")

    # 2. BOTTOM WALL - Constant heat flux
    bottom_boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Xvelocity": 0.0,
            "Yvelocity": 0.0,
            "Temperature__y": -0.026,
        },
        batch_size=cfg.batch_size.boundary // 4,
        criteria=bottom_wall_criteria,
        lambda_weighting={
            "Xvelocity": 1.0,
            "Yvelocity": 1.0,
            "Temperature__y": 1.0
        }
    )
    domain.add_constraint(bottom_boundary, "bottom_wall")

    # 3. RIGHT WALL - Open hydrostatic boundary
    right_boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Temperature": geothermal_temp_expr,  # Function of coordinates
            # Remove pressure constraint for now until you add pressure to network
        },
        batch_size=cfg.batch_size.boundary // 4,
        criteria=right_wall_criteria,
        lambda_weighting={
            "Temperature": 1.0,
        }
    )
    domain.add_constraint(right_boundary, "right_wall")

    # 4. TOP WALL - Fixed temperature
    top_boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Temperature": 20.0,
            "Xvelocity": 0.0,
            "Yvelocity": 0.0,
        },
        batch_size=cfg.batch_size.boundary // 4,
        criteria=top_wall_criteria,
        lambda_weighting={
            "Temperature": 1.0,
            "Xvelocity": 1.0,
            "Yvelocity": 1.0
        }
    )
    domain.add_constraint(top_boundary, "top_wall")


    # --- Interior PDE Constraints ---
    # This enforces the PDE equations throughout the interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "continuity": 0.0,      # Mass conservation
            "heat_equation": 0.0,   # Heat equation
        },
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "continuity": 10.0,
            "heat_equation": 12.0,
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
    viz_times = np.array([0, 900, 1800, 2700, 3600], dtype=float)
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
    plotter = EnhancedChamberPlotter()
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

if __name__ == "__main__":
    create_enhanced_solver()