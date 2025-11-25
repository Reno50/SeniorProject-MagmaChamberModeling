from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE

class GeothermalSystemPDE(PDE):
    '''
    Geothermal / hydrothermal multiphase PDE system with Darcy + advective energy terms.

    Conventions:
    - x: horizontal
    - y: vertical (positive up)
    - q_w, q_s are volumetric fluxes (m/s) computed from Darcy's law
    - mass conservation uses rho * q fluxes
    - energy equation uses conductive + advective enthalpy fluxes
    '''

    name = "GeothermalSystem"

    def __init__(self, expose_velocities: bool = True):
        # independent vars
        time, x, y = Symbol("time"), Symbol("x"), Symbol("y")

        Lx = 20000.0   # chamber width in meters (20 km)
        Ly = 6000.0    # chamber height in meters (6 km)
        # OLD --- t_scale = 300000 * 365.0 * 24.0 * 3600.0  # seconds in 300,000 years
        # NEW --- now, we know that time is scaled by a factor of 1000000 years to 1.0 network input
        t_scale = 1000000 * 365.0 * 24.0 * 3600.0 # seconds in 1000000 years
        temp_scale = 1000 # 1000 degrees in every 1.0 network inputyear

        # Scaling stuff
        dx_factor = 1.0 / Lx
        dy_factor = 1.0 / Ly
        dx2_factor = 1.0 / (Lx * Lx)
        dy2_factor = 1.0 / (Ly * Ly)
        dt_factor = 1.0 / t_scale

        # Anytime we have an equation representing something physical with a .diff(x) or .diff(y), it'll have a * dx_factor or *dy_factor now

        # primary field functions (depend on time,x,y)
        T_phys = Function("Temperature")(time, x, y)
        T = T_phys * temp_scale  # Use scaled temperature everywhere below
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
        rho_s = 600.0       # kg/m^3 (placeholder)
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

        # Darcy volumetric flux q = - (k_perm * k_rel / mu) * grad( p + rho * g * y )
        # note: grad( rho*g*y ) = rho*g in y direction => accounting for gravity
        # compute prefactors
        pref_w = k_perm * k_rw / mu_w
        pref_s = k_perm * k_rs / mu_s

        # volumetric flux components for water (q_w_x, q_w_y) and steam
        q_w_x = -pref_w * p_w.diff(x) * dx_factor
        q_w_y = -pref_w * (p_w.diff(y) * dy_factor + rho_w * g * dy_factor)  # gravity term must also be scaled

        q_s_x = -pref_s * p_s.diff(x) * dx_factor
        q_s_y = -pref_s * (p_s.diff(y) * dy_factor + rho_s * g * dy_factor)  # gravity term must also be scaled

        # mass flux components (rho * q)
        rhoq_w_x = rho_w * q_w_x
        rhoq_w_y = rho_w * q_w_y

        rhoq_s_x = rho_s * q_s_x
        rhoq_s_y = rho_s * q_s_y

        # divergence of mass flux for each phase
        div_rhoq_w = rhoq_w_x.diff(x) * dx_factor + rhoq_w_y.diff(y) * dy_factor
        div_rhoq_s = rhoq_s_x.diff(x) * dx_factor + rhoq_s_y.diff(y) * dy_factor

        # mass storage term
        mass_storage = phi * (rho_w * S_w + rho_s * S_s)

        # MASS: d/dt(mass_storage) + div(rho*q) - q_sf = 0
        mass_eq = mass_storage.diff(time) * dt_factor + (div_rhoq_w + div_rhoq_s) - q_sf

        # conduction divergence: ∇·(-K_a ∇T) = -K_a * Laplacian(T)
        conduction_div = - (K_a * T.diff(x, 2) * dx2_factor + K_a * T.diff(y, 2) * dy2_factor)

        # advective enthalpy fluxes: φ * ρ * h * T * q  (flux components)
        # Note: h_* are treated as specific heat capacities (J/kg/K) for sensible
        # heat so we must multiply by Temperature (T) to get energy per mass.
        adv_w_x = phi * (rho_w * h_w * T * q_w_x)
        adv_w_y = phi * (rho_w * h_w * T * q_w_y)
        adv_s_x = phi * (rho_s * h_s * T * q_s_x)
        adv_s_y = phi * (rho_s * h_s * T * q_s_y)

        adv_div = adv_w_x.diff(x) * dx_factor + adv_w_y.diff(y) * dy_factor + adv_s_x.diff(x) * dx_factor + adv_s_y.diff(y) * dy_factor

        # ENERGY: d/dt(energy_storage) + div( -K_a * grad(T) ) + div( advective_enthalpy ) - q_sh = 0
        # where conduction_div = -K_a * Laplacian(T)
        # Energy per unit volume: include temperature for sensible heat
        energy_storage = phi*(rho_w*h_w*S_w*T + rho_s*h_s*S_s*T) + (1-phi)*rho_r*h_r*T
        energy_eq = energy_storage.diff(time) * dt_factor + conduction_div + adv_div - q_sh

        self.equations = {}
        self.equations["mass_conservation"] = mass_eq
        self.equations["energy_conservation"] = energy_eq

        # Enforce Xv - q_total_x = 0, Yv - q_total_y = 0
        # where q_total = q_w + q_s (total volumetric flux)
        q_total_x = q_w_x + q_s_x
        q_total_y = q_w_y + q_s_y
        # Darcy residuals for velocity outputs
        self.equations["darcy_x"] = Xv - q_total_x
        self.equations["darcy_y"] = Yv - q_total_y

        # To ensure the saturation of steam + water = 1
        self.equations["sat_sum"] = S_s + S_w - 1

        # Add a Neumann boundary condition node for heat flux
        # This allows you to constrain the heat flux directly
        T = Function("Temperature")(Symbol("time"), Symbol("x"), Symbol("y"))
        K_a = 2.5  # thermal conductivity
        
        # Heat flux in y-direction: q_y = -K_a * dT/dy
        # For bottom boundary with upward flux of 0.065 W/m²:
        # -K_a * dT/dy = 0.065 => dT/dy = -0.065/2.5 = -0.026 K/m
        # In normalized coordinates (6 km -> 1): dT/dy_norm = -0.026 * 6000 = -156
        
        self.equations["heat_flux_y"] = -K_a * T.diff(Symbol("y")) * dy_factor
        self.equations["heat_flux_x"] = -K_a * T.diff(Symbol("x")) * dx_factor

        # Notes:
        # - Replace constant rho_s, h_* with temperature/pressure dependent functions for higher fidelity.
        # - Units must be consistent. Ensure q_sf and q_sh have correct units