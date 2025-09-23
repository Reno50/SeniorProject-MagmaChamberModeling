from Solvers.BasicChamberSolver import create_enhanced_solver
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
import physicsnemo.sym

@physicsnemo.sym.main(config_path="conf", config_name="config")
def main(cfg: PhysicsNeMoConfig):
    create_enhanced_solver(cfg)

if __name__ == "__main__":
    main()