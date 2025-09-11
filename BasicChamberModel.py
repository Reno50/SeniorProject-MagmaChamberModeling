# docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia --rm -v ./:/workspace/SeniorProj -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash
from sympy import Symbol, Function
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
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter

class ChamberPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        # only plot x,y dimensions
        invar = {k: v for k, v in invar.items() if k in ["x", "y"]}
        fs = super().__call__(invar, true_outvar, pred_outvar)
        return fs

class MagmaChamberPDE(PDE):
    name = "MagmaChamber"
    
    def __init__(self):
        time, x, y = Symbol('time'), Symbol('x'), Symbol('y')

        # --- Field Variables --- 
        Temperature = Function("Temperature") # Temperature of the magma
        Xvelocity = Function("Xvelocity") # Xvelocity of the magma
        Yvelocity = Function("Yvelocity") # Yvelocity of the magma

        # --- Equations ---

        self.equations = {}
        self.equations["mass_conservation"] = Xvelocity.diff(x) + Yvelocity.diff(y)

@physicsnemo.sym.main(config_path="conf", config_name="config")
def create_solver(cfg: PhysicsNeMoConfig):
    chamber = Rectangle(
        point_1=(0, 0), point_2=(10000,5000), # Using Km as units
        parameterization=Parameterization({Parameter("time"): 0.0})
    )

    time = Symbol("time")
    time_range = {time: (0, 86400)} # 1 day

    network = instantiate_arch(
        input_keys=[Key("time"), Key("x"), Key("y")],
        output_keys=[Key("Temperature"), Key("Xvelocity"), Key("Yvelocity")],
        cfg=cfg.arch.fully_connected,
    )
    magma_pde = MagmaChamberPDE()

    nodes = magma_pde.make_nodes() + [network.make_node(name="magma_net")]

    # --- Domain stuff ---
    domain = Domain()

    boundary = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=chamber,
        outvar={"Temperature": 1200.0, "Xvelocity": 0.0, "Yvelocity": 0.0},
        batch_size=cfg.batch_size.boundary,
        lambda_weighting={"Temperature": 1.0, "Xvelocity": 1.0, "Yvelocity": 1.0}
    )
    domain.add_constraint(boundary, "boundary")
    init_points = chamber.sample_interior(512) # arbitrary sample

    n_val = int(np.asarray(init_points["x"]).shape[0])

    true_init = {
        "Temperature": np.full(n_val, 1200.0, dtype=float), 
        "Xvelocity": np.full(n_val, 0.0, dtype=float), 
        "Yvelocity": np.full(n_val, 0.0, dtype=float)
    }
    
    plotter = ValidatorPlotter()
    
    validator = PointwiseValidator(
        nodes=nodes, invar=init_points, true_outvar=true_init, batch_size=128, plotter=plotter
    )
    domain.add_validator(validator)

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    create_solver()