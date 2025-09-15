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
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import matplotlib.pyplot as plt # Because even the official examples use this

class ChamberPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        # only plot x,y dimensions

        #times = np.unique(invar["time"])
        figures = []

        #for t in times:
        # mask = (invar["time"][:,0] == t) # Each point has an associated time
        x, y = invar["x"][:,0], invar["y"][:,0]

        x,y = invar["x"][:,0], invar["y"][:,0]
        extent = (x.min(), x.max(), y.min(), y.max())

        # output variables
        temp_true = true_outvar["Temperature"]
        temp_pred = pred_outvar["Temperature"]

        temp_true, temp_pred = self.interpolate_output(x, y, [temp_true, temp_pred], extent)

        f = plt.figure(figsize=(14,4), dpi=100)
        plt.suptitle(f"Lava chamber at t={t/3600:.1f} hours")
        plt.subplot(1, 2, 1)
        plt.title("True")
        plt.imshow(temp_true.T, origin="lower", extent=extent)
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Predicted")
        plt.imshow(temp_pred.T, origin="lower", extent=extent)
        plt.colorbar()

        figures.append((f, f"temp_t{int(0)}"))
        return figures
    
    @staticmethod
    def interpolate_output(x, y, us, extent):
        "Interpolates irregular points onto a mesh"

        # define mesh to interpolate onto
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100),
            np.linspace(extent[2], extent[3], 100),
            indexing="ij",
        )

        # linearly interpolate points onto mesh
        us = [scipy.interpolate.griddata( (x, y), u, tuple(xyi) ) for u in us]
        return us

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
        parameterization=Parameterization({Parameter("time"): (0.0, 86400.0)})
    )

    time = Symbol("time")

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

    ## times to visualize (0h, 6h, 12h, 24h)
    #times = np.array([0, 6*3600, 12*3600, 24*3600])

    init_points = chamber.sample_interior(512) # arbitrary sample

    invar = {}
    for k, v in init_points.items():
        invar[k] = np.tile(v, 1)

    # n_val = int(np.asarray(init_points["x"]).shape[0])

    #invar["time"] = np.repeat(times, init_points["x"].shape[0])[:, None]

    n_val = invar["x"].shape[0]

    true_init = {
        "Temperature": np.full((n_val, 1), 1200.0, dtype=float), 
        "Xvelocity": np.full((n_val, 1), 0.0, dtype=float), 
        "Yvelocity": np.full((n_val, 1), 0.0, dtype=float)
    }
    
    plotter = ChamberPlotter()

    validator = PointwiseValidator(
        nodes=nodes, invar=init_points, true_outvar=true_init, batch_size=128, plotter=plotter
    )
    domain.add_validator(validator)

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    create_solver()