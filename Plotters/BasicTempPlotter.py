import matplotlib.pyplot as plt
import numpy as np
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import scipy

class ChamberPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        """Plot results at multiple time steps"""

        timeScalingFactor = 1000000.0 # 1.0 time in the neural network is 1000 kyrs, so 800,000 years will be 0.8 in the network
        # Also defined in the solver file
        tempScalingFactor = 1000.0 # 1000 degrees is 1.0 in the network, similar to time

        # Get unique time values
        times = np.unique(invar["time"][:,0])
        chamber_width, chamber_height = 20000, 6000 # Same for chamber size - normalize to 0 - 1
        figures = []

        min_true_temp = max(true_outvar["Temperature"].min(), -0.00001) # If scale goes past 0, there is a major problem, but now you can at least see that it is problematic
        min_pred_temp = min(pred_outvar["Temperature"].min(), -0.00001)
            
        max_true_temp = max(true_outvar["Temperature"].max() * tempScalingFactor, 901) # Same idea here
        max_pred_temp = min(pred_outvar["Temperature"].max() * tempScalingFactor, 901)

        for t in times:
            # Filter data for this specific time
            time_mask = (invar["time"][:,0] == t)
            x = invar["x"][time_mask,0]
            y = invar["y"][time_mask,0]
            
            if len(x) == 0:  # Skip if no data for this time
                continue
                
            extent = (x.min(), x.max(), y.min(), y.max())

            # Get temperature data for this time step
            temp_true = true_outvar["Temperature"][time_mask] * tempScalingFactor # Because initial conditions are also in squashed temps
            temp_pred = pred_outvar["Temperature"][time_mask] * tempScalingFactor
            
            # Interpolate onto regular grid
            temp_true_interp, temp_pred_interp = self.interpolate_output(
                x, y, [temp_true, temp_pred], extent
            )

            # Create figure
            f = plt.figure(figsize=(16, 6), dpi=100)
            plt.suptitle(f"Magma Chamber at {t*timeScalingFactor:.1f} years", fontsize=16)
            
            # Flip Y limits for plotting so (0,0) is top-left
            plot_extent = (extent[0], extent[1], extent[3], extent[2])

            # True temperature
            plt.subplot(1, 3, 1)
            plt.title("True Temperature")
            im1 = plt.imshow(temp_true_interp, origin="upper", extent=plot_extent, 
                           cmap='hot', vmin=min_true_temp, vmax=max_true_temp)
            plt.colorbar(im1, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Predicted temperature
            plt.subplot(1, 3, 2)
            plt.title("Predicted Temperature")
            im2 = plt.imshow(temp_pred_interp, origin="upper", extent=plot_extent, 
                           cmap='hot', vmin=min_pred_temp, vmax=max_pred_temp)
            plt.colorbar(im2, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")

            plt.tight_layout()
            figures.append((f, f"temp_t{t*timeScalingFactor:.1f} years"))
            
        return figures
    
    @staticmethod
    def interpolate_output(x, y, us, extent):
        """Interpolates irregular points onto a (y, x) mesh so imshow displays correctly"""
        
        # Create the grid with Y first (rows) and X second (columns)
        yi = np.linspace(extent[2], extent[3], 100)  # ymin → ymax
        xi = np.linspace(extent[0], extent[1], 100)  # xmin → xmax

        Y, X = np.meshgrid(yi, xi, indexing="ij")  # shape: [ny, nx]

        # Interpolate each variable; output matches Y,X grid shape
        out = [
            scipy.interpolate.griddata(
                (x, y),
                u.ravel(),
                (X, Y),
                method="nearest",
                fill_value=np.nan
            )
            for u in us
        ]

        return out