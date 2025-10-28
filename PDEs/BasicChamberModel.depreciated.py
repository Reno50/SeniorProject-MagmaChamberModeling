from physicsnemo.sym.eq.pde import PDE
from sympy import Symbol, Function

class BasicMagmaChamberPDE(PDE):
    """
    PDE system for magma chamber modeling including:
    - Mass conservation (continuity equation)
    - Heat equation with convection and diffusion
    - Simplified momentum (Stokes flow for high-viscosity magma)
    """
    name = "BasicMagmaChamber"
    
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