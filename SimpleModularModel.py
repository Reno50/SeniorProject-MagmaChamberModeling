from Solvers.BasicChamberSolver import create_enhanced_solver
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
import physicsnemo.sym

# Run with docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --runtime nvidia --rm -v ./:/workspace -it nvcr.io/nvidia/physicsnemo/physicsnemo:25.06 bash

# Then do python SimpleModularModel.py

@physicsnemo.sym.main(config_path="conf", config_name="config")
def main(cfg: PhysicsNeMoConfig):
    create_enhanced_solver(cfg)

if __name__ == "__main__":
    main()