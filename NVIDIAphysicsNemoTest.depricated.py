# =============================================================================
# 4. NVIDIA physicsnemo - Advanced Coupled System
# =============================================================================

# This would be a more complex implementation in NVIDIA physicsnemo
# showing the coupled thermal-mechanical-chemical system

import torch as nn
import numpy as np
from sympy import Symbol, Eq, tanh, And, Or, Function
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.key import Key
from physicsnemo.sym.models.fully_connected import FullyConnectedArch
from physicsnemo.sym.node import Node

class MagmaChamberPDE(PDE):
    name = "MagmaChamber"
    
    def __init__(self):
        # Define symbols
        time, x, y = Symbol('t'), Symbol('x'), Symbol('y')
        
        # Field variables
        T = Function('T')(time, x, y)      # Temperature
        u = Function('u')(time, x, y)      # x-velocity  
        v = Function('v')(time, x, y)      # y-velocity
        p = Function('p')(time, x, y)      # Pressure
        C = Function('C')(time, x, y)      # Chemical concentration
        
        # Physical parameters
        rho = Symbol('rho')      # Density
        mu = Symbol('mu')        # Viscosity
        cp = Symbol('cp')        # Heat capacity
        k_thermal = Symbol('k')  # Thermal conductivity
        D = Symbol('D')          # Chemical diffusivity
        alpha = Symbol('alpha')  # Thermal expansion
        g = Symbol('g')          # Gravity
        
        # Input variables
        input_variables = {'t': time, 'x': x, 'y': y}
        
        # Equations
        # Heat equation with convection
        heat_eq = (
            rho * cp * (T.diff(time) + u * T.diff(x) + v * T.diff(y)) -
            k_thermal * (T.diff(x, 2) + T.diff(y, 2))
        )
        
        # Stokes flow with thermal buoyancy
        momentum_x = (
            mu * (u.diff(x, 2) + u.diff(y, 2)) - p.diff(x)
        )
        
        momentum_y = (
            mu * (v.diff(x, 2) + v.diff(y, 2)) - p.diff(y) +
            rho * g * alpha * (T - 1200)  # Buoyancy term
        )
        
        # Continuity
        continuity = u.diff(x) + v.diff(y)
        
        # Chemical advection-diffusion
        chemical_eq = (
            C.diff(time) + u * C.diff(x) + v * C.diff(y) -
            D * (C.diff(x, 2) + C.diff(y, 2))
        )
        
        # Create equations dictionary
        self.equations = {}
        self.equations['heat_equation'] = heat_eq
        self.equations['momentum_x'] = momentum_x  
        self.equations['momentum_y'] = momentum_y
        self.equations['continuity'] = continuity
        self.equations['chemical_transport'] = chemical_eq

def create_magma_chamber_solver():
    # Geometry
    chamber = Rectangle(
        point_1=(0, 0), point_2=(10000, 5000)  # 10km x 5km chamber
    )
    
    # Time domain
    t_symbol = Symbol('t')
    time_range = {t_symbol: (0, 86400)}  # 1 day simulation
    
    # Create nodes
    flow_net = FullyConnectedArch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("T"), Key("u"), Key("v"), Key("p"), Key("C")],
        nr_layers=4, layer_size=512
    )


    magma_pde = MagmaChamberPDE()
    # flow_net = Node.from_torch_module(
    #     torch.nn.Sequential(
    #         torch.nn.Linear(3, 512),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(512, 512), 
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(512, 512),
    #         torch.nn.Tanh(),
    #         torch.nn.Linear(512, 5)  # T, u, v, p, C
    #     ),
    #     inputs=['t', 'x', 'y'],
    #     outputs=['T', 'u', 'v', 'p', 'C']
    # )
    
    nodes = magma_pde.make_nodes() + [flow_net.make_node(name="flow_net")]

    params = {
        Symbol('rho'): 2700.,   # kg/m^3 (example)
        Symbol('mu'): 1e18,     # Pa*s (example)
        Symbol('cp'): 1000.,
        Symbol('k'): 2.0,
        Symbol('D'): 1e-9,
        Symbol('alpha'): 3e-5,
        Symbol('g'): 9.81
    }

    # When constructing constraints, use parameterization merging time_range & params
    parameterization_var = { Symbol('t'): (0, 86400), **params }
    
    # Constraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={
            'heat_equation': 0,
            'momentum_x': 0,
            'momentum_y': 0, 
            'continuity': 0,
            'chemical_transport': 0
        },
        batch_size=2000,
        parameterization=parameterization_var
    )
    
    # Boundary conditions
    walls = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={'u': 0, 'v': 0},  # No-slip
        batch_size=500,
        parameterization=time_range
    )
    
    # Domain
    domain = Domain()
    domain.add_constraint(interior, "interior")
    domain.add_constraint(walls, "walls")
    
    # Solver
    slv = Solver(domain, nodes)
    
    return slv

# Usage
solver = create_magma_chamber_solver()
solver.solve()
