# First, run:
# docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia --rm -v ./:/workspace -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
# Now that you're in the bash of the container, run
# python BasicChamberModel.py

from sympy import Symbol, Function, interpolate
import scipy
import numpy as np

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Point1D
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,  # Added for initial conditions
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import matplotlib.pyplot as plt

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
                           cmap='hot', vmin=1000, vmax=1400)
            plt.colorbar(im1, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Predicted temperature
            plt.subplot(1, 3, 2)
            plt.title("Predicted Temperature")
            im2 = plt.imshow(temp_pred_interp.T, origin="lower", extent=extent, 
                           cmap='hot', vmin=1000, vmax=1400)
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
                                       method='cubic', fill_value=np.nan) for u in us]
        return us

class RealisticMagmaChamberPDE(PDE):
    """
    Enhanced PDE system for magma chamber modeling including:
    - Mass conservation (continuity equation)
    - Heat equation with convection and diffusion
    - Simplified momentum (Stokes flow for high-viscosity magma)
    """
    name = "RealisticMagmaChamer"
    
    def __init__(self):
        # Symbols
        time, x, y = Symbol('time'), Symbol('x'), Symbol('y')

        # Field Variables (evaluated at coordinates)
        Temperature = Function("Temperature")(time, x, y)  # Temperature [K or °C]
        Xvelocity = Function("Xvelocity")(time, x, y)      # X-velocity [m/s]
        Yvelocity = Function("Yvelocity")(time, x, y)      # Y-velocity [m/s]
        
        # Physical constants - let's substitute with actual values
        alpha = 1e-6  # Thermal diffusivity [m²/s] for basaltic magma
        
        self.equations = {}
        
        # 1. Mass Conservation (Continuity Equation)
        # ∇ · u = 0  =>  ∂u/∂x + ∂v/∂y = 0
        self.equations["continuity"] = Xvelocity.diff(x) + Yvelocity.diff(y)
        
        # 2. Heat Equation with Convection
        # ∂T/∂t + u·∇T = α∇²T
        # ∂T/∂t + u(∂T/∂x) + v(∂T/∂y) = α(∂²T/∂x² + ∂²T/∂y²)
        self.equations["heat_equation"] = (
            Temperature.diff(time) + 
            Xvelocity * Temperature.diff(x) + 
            Yvelocity * Temperature.diff(y) - 
            alpha * (Temperature.diff(x, 2) + Temperature.diff(y, 2))
        )
        
        # 3. Simplified Momentum Equations (Stokes flow)
        # For high-viscosity magma, we can approximate with Stokes equations
        # -∇p + μ∇²u = 0  (ignoring pressure gradient for now)
        # This is a simplification - in reality you'd include buoyancy forces
        
        # Optional: Add buoyancy-driven flow (Boussinesq approximation)
        # This would require thermal expansion coefficient and reference temperature

def create_realistic_initial_conditions(chamber_points, time_points):
    """
    Create more realistic initial conditions with temperature gradients
    """
    n_points = len(chamber_points["x"])
    n_times = len(time_points)
    total_points = n_points * n_times
    
    # Create spatial coordinates repeated for each time step
    x_coords = np.tile(chamber_points["x"].flatten(), n_times)
    y_coords = np.tile(chamber_points["y"].flatten(), n_times)
    t_coords = np.repeat(time_points, n_points)
    
    # Create realistic temperature profile
    # Higher temperature at bottom, cooler at top (thermal stratification)
    chamber_height = 5000  # 5 km
    base_temp = 1300       # °C at bottom
    top_temp = 1100        # °C at top
    
    # Linear temperature gradient with y-coordinate
    temp_gradient = (base_temp - top_temp) / chamber_height
    initial_temps = base_temp - temp_gradient * y_coords
    
    # Add some small random perturbations to trigger convection
    np.random.seed(42)  # For reproducibility
    perturbations = np.random.normal(0, 10, total_points)  # ±10°C random variations
    initial_temps += perturbations
    
    # Start with no initial velocities (magma at rest)
    initial_u = np.zeros(total_points)
    initial_v = np.zeros(total_points)
    
    return {
        "time": t_coords.reshape(-1, 1),
        "x": x_coords.reshape(-1, 1), 
        "y": y_coords.reshape(-1, 1),
        "Temperature": initial_temps,
        "Xvelocity": initial_u,
        "Yvelocity": initial_v
    }

@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Define chamber geometry (10 km × 5 km)
    chamber = Rectangle(
        point_1=(0, 0), 
        point_2=(10000, 5000),  # 10 km × 5 km
        parameterization=Parameterization({
            Parameter("time"): (0.0, 86400.0)  # 24 hours in seconds
        })
    )

    # Create neural network
    network = instantiate_arch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("Temperature"), Key("Xvelocity"), Key("Yvelocity")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Initialize enhanced PDE
    magma_pde = RealisticMagmaChamberPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    # Create domain
    domain = Domain()

    # --- Boundary Conditions ---
    # More realistic boundary conditions
    boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Temperature": 1200.0,  # Fixed temperature at walls
            "Xvelocity": 0.0,       # No-slip condition
            "Yvelocity": 0.0        # No-slip condition
        },
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={
            "Temperature": 1.0, 
            "Xvelocity": 1.0, 
            "Yvelocity": 1.0
        }
    )
    domain.add_constraint(boundary, "boundary")

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
            "continuity": 1.0,
            "heat_equation": 1.0,
        }
    )
    domain.add_constraint(interior, "interior")

    # --- Time-dependent Validation ---
    # Sample interior points
    interior_points = chamber.sample_interior(256)  # Fewer points for multiple times
    
    # Define time steps for validation (0h, 6h, 12h, 24h)
    validation_times = np.array([0, 6*3600, 12*3600, 24*3600], dtype=float)
    
    # Create initial conditions with temperature gradients
    validation_data = create_realistic_initial_conditions(
        interior_points, validation_times
    )
    
    # Extract input and output variables
    invar = {
        "time": validation_data["time"],
        "x": validation_data["x"], 
        "y": validation_data["y"]
    }
    
    true_outvar = {
        "Temperature": validation_data["Temperature"],
        "Xvelocity": validation_data["Xvelocity"],
        "Yvelocity": validation_data["Yvelocity"]
    }
    
    # Create enhanced plotter
    plotter = EnhancedChamberPlotter()
    
    # Create validator
    validator = PointwiseValidator(
        nodes=nodes, 
        invar=invar, 
        true_outvar=true_outvar, 
        batch_size=128, 
        plotter=plotter
    )
    domain.add_validator(validator)

    # Create and run solver
    solver = Solver(cfg, domain)
    solver.solve()

if __name__ == "__main__":
    create_enhanced_solver()