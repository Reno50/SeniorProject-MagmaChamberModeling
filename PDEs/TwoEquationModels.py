from sympy import Symbol, Function
from physicsnemo.sym.eq.pde import PDE

class HydrothermMimicPDE(PDE):
    """
    A PDE system made from scratch to replicate the paper's Hydrotherm equations
    Using the Ground-Water Flow Equation and the Thermal-Energy Transport Equation
    """
    def __init__(self):
        time, x, y = Symbol('time'), Symbol('x'), Symbol('y')

        Temperature = Function("Temperature")(time, x, y)

        '''
        The equation is made up of 4 terms
        First, (Porosity * (Water_Density*Saturation_Of_Water + Steam_Density*Saturation_Of_Water_In_Steam)).diff(time)
        Plus, 
        '''

from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE

class GeothermalSystemPDE(PDE):
    """
    Geothermal / hydrothermal multiphase PDE system with Darcy + advective energy terms.

    Conventions:
    - x: horizontal
    - y: vertical (positive up)
    - q_w, q_s are volumetric fluxes (m/s) computed from Darcy's law
    - mass conservation uses rho * q fluxes
    - energy equation uses conductive + advective enthalpy fluxes
    """

    name = "GeothermalSystem"

    def __init__(self, expose_velocities: bool = True):
        # independent vars
        time, x, y = Symbol("time"), Symbol("x"), Symbol("y")

        # primary field functions (depend on time,x,y)
        T = Function("Temperature")(time, x, y)
        p_w = Function("Pressure_water")(time, x, y)
        p_s = Function("Pressure_steam")(time, x, y)
        S_w = Function("Saturation_water")(time, x, y)
        S_s = Function("Saturation_steam")(time, x, y)

        Xv = Function("XVelocity")(time, x, y)  # volumetric flux x-component (m/s)
        Yv = Function("YVelocity")(time, x, y)  # volumetric flux y-component (m/s)

        # --------------------------------------------------------------------------------
        # physical constants (tune / replace with functions later)
        phi = 0.1           # porosity
        rho_w = 1000.0      # kg/m^3
        rho_s = 600         # kg/m^3 (placeholder)
        rho_r = 2700.0      # rock density kg/m^3
        k_perm = 1e-13      # permeability m^2
        k_rw = 1.0          # rel perm water
        k_rs = 1.0          # rel perm steam
        mu_w = 1e-3         # water viscosity Pa·s
        mu_s = 1e-5         # steam viscosity Pa·s
        g = 9.81            # m/s^2

        # Enthalpies (ideally temperature-dependent functions)
        h_w = 4186.0        # J/(kg K) (approx cp for water)
        h_s = 2676000.0     # J/kg latent + sensible (placeholder)
        h_r = 1000.0        # J/(kg K) rock / placeholder

        # thermal conductivity
        K_a = 2.5           # W/(m K)

        # source terms (set to 0 by default, replace with functions if needed)
        q_sf = 0            # mass source (kg/(m^3 s) or consistent units depending on formulation)
        q_sh = 0            # heat source (W/m^3)

        # --------------------------------------------------------------------------------
        # Darcy volumetric flux q = - (k_perm * k_rel / mu) * grad( p + rho * g * y )
        # note: grad( rho*g*y ) = rho*g in y direction => accounting for gravity
        # compute prefactors
        pref_w = k_perm * k_rw / mu_w
        pref_s = k_perm * k_rs / mu_s

        # volumetric flux components for water (q_w_x, q_w_y) and steam
        q_w_x = -pref_w * p_w.diff(x)
        q_w_y = -pref_w * (p_w.diff(y) + rho_w * g)  # includes gravity term in y

        q_s_x = -pref_s * p_s.diff(x)
        q_s_y = -pref_s * (p_s.diff(y) + rho_s * g)

        # mass flux components (rho * q)
        rhoq_w_x = rho_w * q_w_x
        rhoq_w_y = rho_w * q_w_y

        rhoq_s_x = rho_s * q_s_x
        rhoq_s_y = rho_s * q_s_y

        # divergence of mass flux for each phase
        div_rhoq_w = rhoq_w_x.diff(x) + rhoq_w_y.diff(y)
        div_rhoq_s = rhoq_s_x.diff(x) + rhoq_s_y.diff(y)

        # mass storage term
        mass_storage = phi * (rho_w * S_w + rho_s * S_s)

        # MASS: d/dt(mass_storage) + div(rho*q) - q_sf = 0
        mass_eq = mass_storage.diff(time) + (div_rhoq_w + div_rhoq_s) - q_sf

        # conduction divergence: ∇·(-K_a ∇T) = -K_a * Laplacian(T)
        conduction_div = - (K_a * T.diff(x, 2) + K_a * T.diff(y, 2))

        # advective enthalpy fluxes: φ * ρ * h * q  (flux components)
        adv_w_x = phi * (rho_w * h_w * q_w_x)
        adv_w_y = phi * (rho_w * h_w * q_w_y)
        adv_s_x = phi * (rho_s * h_s * q_s_x)
        adv_s_y = phi * (rho_s * h_s * q_s_y)

        adv_div = adv_w_x.diff(x) + adv_w_y.diff(y) + adv_s_x.diff(x) + adv_s_y.diff(y)

        # ENERGY: d/dt(energy_storage) + div( -K_a * grad(T) ) + div( advective_enthalpy ) - q_sh = 0
        # where conduction_div = -K_a * Laplacian(T)
        energy_storage = phi*(rho_w*h_w*S_w + rho_s*h_s*S_s) + (1-phi)*rho_r*h_r
        energy_eq = energy_storage.diff(time) + conduction_div + adv_div - q_sh

        # --------------------------------------------------------------------------------
        # build equations dict
        self.equations = {}
        self.equations["mass_conservation"] = mass_eq
        self.equations["energy_conservation"] = energy_eq

        # --------------------------------------------------------------------------------
        # Optionally expose Darcy residuals so network must learn velocities consistent with pressure:
        # If velocities are outputs, enforce Xv - q_total_x = 0, Yv - q_total_y = 0
        # where q_total = q_w + q_s (total volumetric flux)
        if expose_velocities:
            q_total_x = q_w_x + q_s_x
            q_total_y = q_w_y + q_s_y
            # Darcy residuals for velocity outputs
            self.equations["darcy_x"] = Xv - q_total_x
            self.equations["darcy_y"] = Yv - q_total_y
        else:
            # if you still want to monitor consistency, you could expose a diagnostic residual (optional)
            # self.equations["darcy_diagnostic_x"] = q_w_x + q_s_x
            # self.equations["darcy_diagnostic_y"] = q_w_y + q_s_y
            pass

        # --------------------------------------------------------------------------------
        # Notes:
        # - Replace constant rho_s, h_* with temperature/pressure dependent functions for higher fidelity.
        # - Units must be consistent. Ensure q_sf and q_sh have correct units for your formulation.
        # - You might want to nondimensionalize or rescale variables (length, time, temperature) for training stability.

"""
class GeothermalSystemPDE(PDE):
    '''
    PDE system based on the ground-water flow and thermal-energy transport equations
    for geothermal/hydrothermal systems with multiphase flow.
    
    This is what the paper mentions they use in Hydrotherm simulations
    '''
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
        Xvelocity = Function("XVelocity")(time, x, y)      # u [m/s] - snatched from the simple model
        Yvelocity = Function("YVelocity")(time, x, y)      # v [m/s] - snatched from the simple model
        
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

        # Momentum equations (Darcy's law for porous medium)
        # u = -(k/μ) * ∂p/∂x
        # v = -(k/μ) * (∂p/∂y + ρg)
        # This one might be wierd - none of these fake things are based on the paper at all

        # I'm gonna replace the Pressure term with:
        fake_pressure = (Pressure_water * Saturation_water) + (Pressure_steam * Saturation_steam)
        # and the mu term (which is viscosity) with a saturation-multiplied term
        fake_viscosity = (Saturation_water * mu_w) + (Saturation_steam * mu_s)
        # and for rho, lets just use water for now I guess :/
        self.equations["darcy_x"] = Xvelocity + (k / fake_viscosity) * fake_pressure.diff(x)
        self.equations["darcy_y"] = Yvelocity + (k / fake_viscosity) * (fake_pressure.diff(y) + rho_w * 9.81)"""

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