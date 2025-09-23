from sympy import Symbol, Function
from physicsnemo.sym.eq.pde import PDE

class GeothermalSystemPDE(PDE):
    """
    PDE system based on the ground-water flow and thermal-energy transport equations
    for geothermal/hydrothermal systems with multiphase flow.
    
    Note: This is quite different from typical magma chamber convection models
    """
    name = "GeothermalSystem"
    
    def __init__(self):
        # Symbols
        time, x, y = Symbol('time'), Symbol('x'), Symbol('y')

        # Primary field variables
        Temperature = Function("Temperature")(time, x, y)  # T [°C or K]
        Pressure_water = Function("Pressure_water")(time, x, y)  # p_w [Pa]
        Pressure_steam = Function("Pressure_steam")(time, x, y)  # p_s [Pa]
        Saturation_water = Function("Saturation_water")(time, x, y)  # S_w [dimensionless]
        Saturation_steam = Function("Saturation_steam")(time, x, y)  # S_s [dimensionless]
        
        # For simplified 2D case, we might also need velocity components
        # derived from pressure gradients via Darcy's law
        
        # Physical parameters (these would typically be spatially variable)
        phi = 0.1           # Porosity [dimensionless]
        rho_w = 1000.0      # Water density [kg/m³]
        rho_s = 0.6         # Steam density [kg/m³] (pressure/temperature dependent)
        rho_r = 2700.0      # Rock density [kg/m³]
        k = 1e-13           # Permeability [m²]
        k_rw = 1.0          # Relative permeability water [dimensionless]
        k_rs = 1.0          # Relative permeability steam [dimensionless]
        mu_w = 1e-3         # Water viscosity [Pa·s]
        mu_s = 1e-5         # Steam viscosity [Pa·s]
        g = 9.81            # Gravity [m/s²]
        
        # Thermal properties
        h_w = 4186          # Specific enthalpy water [J/kg] (temperature dependent)
        h_s = 2676000       # Specific enthalpy steam [J/kg] (temperature dependent)
        h_r = 1000          # Specific enthalpy rock [J/kg]
        K_a = 2.5           # Effective thermal conductivity [W/m·°C]
        
        self.equations = {}
        
        # Equation (36): Ground-Water Flow Equation
        # ∂/∂t[φ(ρ_w S_w + ρ_s S_s)] - ∇·(k k_rw ρ_w/μ_w) - ∇·(k k_rs ρ_s/μ_s)[∇p_g + ρ_s g ê_z] - q_sf = 0
        
        # Simplified mass conservation (assuming single phase for now)
        # This is a major simplification - the full equation requires pressure-saturation relationships
        mass_storage = phi * (rho_w * Saturation_water + rho_s * Saturation_steam)
        
        # Darcy flow terms (simplified - should include pressure gradients)
        # For full implementation, you'd need:
        # water_flow = k * k_rw * rho_w / mu_w * gradient(Pressure_water + rho_w * g * y)
        # steam_flow = k * k_rs * rho_s / mu_s * gradient(Pressure_steam + rho_s * g * y)
        
        self.equations["mass_conservation"] = mass_storage.diff(time)
        # Note: This is incomplete - needs flow terms and source terms
        
        # Equation (37): Thermal-Energy Transport Equation  
        # ∂/∂t[φ(ρ_w h_w S_w + ρ_s h_s S_s + (1-φ)ρ_r h_r)] - ∇·K_a I∇T + ∇·φ(S_w ρ_w h_w V_w + S_s ρ_s h_s V_s) - q_sh = 0
        
        # Energy storage term
        energy_storage = (phi * (rho_w * h_w * Saturation_water + rho_s * h_s * Saturation_steam) + 
                         (1 - phi) * rho_r * h_r)
        
        # Conductive heat transfer
        heat_conduction = K_a * (Temperature.diff(x, 2) + Temperature.diff(y, 2))
        
        # Advective heat transfer (simplified - needs velocity fields)
        # heat_advection = phi * (S_w * rho_w * h_w * V_w + S_s * rho_s * h_s * V_s)
        
        self.equations["energy_conservation"] = (
            energy_storage.diff(time) - heat_conduction
            # + heat_advection - q_sh  # Missing terms
        )

# Alternative: Simplified single-phase version more suitable for magma chambers
class SimplifiedMagmaChamberPDE(PDE):
    """
    Simplified version focusing on temperature and single-phase flow,
    more appropriate for magma chamber modeling
    """
    name = "SimplifiedMagmaChamber"
    
    def __init__(self):
        # Symbols
        time, x, y = Symbol('time'), Symbol('x'), Symbol('y')

        # Field variables
        Temperature = Function("Temperature")(time, x, y)  # T [°C]
        Pressure = Function("Pressure")(time, x, y)        # p [Pa]
        Xvelocity = Function("Xvelocity")(time, x, y)      # u [m/s]
        Yvelocity = Function("Yvelocity")(time, x, y)      # v [m/s]
        
        # Physical constants for magma
        rho = 2600.0        # Magma density [kg/m³]
        cp = 1200.0         # Specific heat [J/kg·K]
        k_thermal = 2.5     # Thermal conductivity [W/m·K]
        alpha = k_thermal / (rho * cp)  # Thermal diffusivity [m²/s]
        
        # For Darcy flow in porous medium (if applicable)
        phi = 0.1           # Porosity
        k_perm = 1e-12      # Permeability [m²]
        mu = 1e3            # Viscosity [Pa·s] (much higher for magma)
        
        self.equations = {}
        
        # Mass conservation (continuity)
        self.equations["continuity"] = Xvelocity.diff(x) + Yvelocity.diff(y)
        
        # Energy conservation (simplified heat equation)
        self.equations["heat_equation"] = (
            Temperature.diff(time) + 
            Xvelocity * Temperature.diff(x) + 
            Yvelocity * Temperature.diff(y) - 
            alpha * (Temperature.diff(x, 2) + Temperature.diff(y, 2))
        )
        
        # Momentum equations (Darcy's law for porous medium)
        # u = -(k/μ) * ∂p/∂x
        # v = -(k/μ) * (∂p/∂y + ρg)
        self.equations["darcy_x"] = Xvelocity + (k_perm / mu) * Pressure.diff(x)
        self.equations["darcy_y"] = Yvelocity + (k_perm / mu) * (Pressure.diff(y) + rho * 9.81)