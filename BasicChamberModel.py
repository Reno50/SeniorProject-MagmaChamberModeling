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
from physicsnemo.sym.geometry.primitives_2d import Rectangle
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
        
        # Physical constants
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

def create_initial_temperature_field(x_coords, y_coords):
    """
    Create realistic initial temperature distribution
    """
    # Convert coordinates from meters to km for easier calculation
    x_km = x_coords / 1000.0
    y_km = y_coords / 1000.0
    
    chamber_height = 5.0  # 5 km
    base_temp = 1300      # °C at bottom
    top_temp = 700       # °C at top
    
    # Create temperature gradient (hotter at bottom)
    temp_gradient = (base_temp - top_temp) / chamber_height
    temp_field = base_temp - temp_gradient * y_km
    
    # Add thermal plume in center-bottom to trigger convection
    center_x = 5.0  # Center of 10km chamber
    plume_strength = 50  # Additional temperature boost
    plume_radius = 1.0   # 1 km radius
    
    # Gaussian thermal plume
    distance_from_plume = np.sqrt((x_km - center_x)**2 + (y_km - 0.5)**2)
    thermal_plume = plume_strength * np.exp(-(distance_from_plume / plume_radius)**2)
    
    # Only add plume in bottom half of chamber
    bottom_mask = y_km < (chamber_height / 2)
    temp_field += thermal_plume * bottom_mask
    
    return temp_field

@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_enhanced_solver(cfg: PhysicsNeMoConfig):
    # Define chamber geometry (10 km × 5 km)
    chamber = Rectangle(
        point_1=(0, 0), 
        point_2=(10000, 5000),  # 10 km × 5 km
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
    magma_pde = RealisticMagmaChamberPDE()
    nodes = magma_pde.make_nodes() + [network.make_node(name="enhanced_magma_net")]

    # Create domain
    domain = Domain()

    # --- Boundary Conditions ---
    # Reasonable boundary conditions
    boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            "Temperature": 1000.0,  # Fixed temperature at walls
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

    interior_t0 = Rectangle(
        point_1=(0, 0), 
        point_2=(10000, 5000),
        parameterization=Parameterization({
            Parameter("time"): 0.0
        })
    )
    
    # Simple interior constraint at t=0 - let network interpolate
    interior_initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_t0,
        outvar={
            "Xvelocity": 0.0,     # Start at rest everywhere
            "Yvelocity": 0.0,     # Start at rest everywhere
        },
        batch_size=256,
        lambda_weighting={
            "Xvelocity": 3.0,
            "Yvelocity": 3.0,
        }
    )
    domain.add_constraint(interior_initial, "initial_velocities")

    # --- Time-dependent Validation ---
    # --- Validation: Only check t=0, let network evolve freely ---
    val_points = chamber.sample_interior(128)

    # Validation at t=0 only (supervised learning for initial conditions)
    val_invar_t0 = {
        "time": np.zeros((128, 1)),  # Only t=0 
        "x": val_points["x"],
        "y": val_points["y"],
    }

    # Create realistic initial temperature at t=0 only
    val_temps_t0 = create_initial_temperature_field(
        val_points["x"].flatten(), 
        val_points["y"].flatten()
    )

    val_outvar_t0 = {
        "Temperature": val_temps_t0,
        "Xvelocity": np.zeros_like(val_temps_t0),
        "Yvelocity": np.zeros_like(val_temps_t0),
    }

    # This validator only checks initial conditions
    validator_t0 = PointwiseValidator(
        nodes=nodes, 
        invar=val_invar_t0, 
        true_outvar=val_outvar_t0, 
        batch_size=64, 
        plotter=None
    )
    domain.add_validator(validator_t0)

    # --- Separate visualization validator (no constraints on evolution) ---
    viz_times = np.array([0, 900, 1800, 2700, 3600], dtype=float)
    viz_points = chamber.sample_interior(64)

    n_viz_times = len(viz_times)
    n_viz_points = viz_points["x"].shape[0]

    viz_invar = {
        "time": np.tile(viz_times, n_viz_points).reshape(-1, 1),
        "x": np.repeat(viz_points["x"], n_viz_times, axis=0),
        "y": np.repeat(viz_points["y"], n_viz_times, axis=0),
    }

    # Dummy values just for plotter compatibility
    dummy_temps = create_initial_temperature_field(
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