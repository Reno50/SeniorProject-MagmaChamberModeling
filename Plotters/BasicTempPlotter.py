import matplotlib.pyplot as plt
import numpy as np
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import scipy

class ChamberPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        """Plot results at multiple time steps"""
        
        # Get unique time values
        times = np.unique(invar["time"][:,0])
        # Endtime = 300kr
        endTime = 300000 # in years
        chamber_width, chamber_height = 20000, 6000 # Same for chamber size - normalize to 0 - 1
        figures = []

        min_true_temp = true_outvar["Temperature"].min()
        min_pred_temp = pred_outvar["Temperature"].min()
            
        max_true_temp = true_outvar["Temperature"].max()
        max_pred_temp = pred_outvar["Temperature"].max()

        for t in times:
            # Filter data for this specific time
            time_mask = (invar["time"][:,0] == t)
            x = invar["x"][time_mask,0]
            y = invar["y"][time_mask,0]
            
            if len(x) == 0:  # Skip if no data for this time
                continue
                
            extent = (x.min(), x.max(), y.min(), y.max())

            # Get temperature data for this time step
            temp_true = true_outvar["Temperature"][time_mask]
            temp_pred = pred_outvar["Temperature"][time_mask]
            
            # Interpolate onto regular grid
            temp_true_interp, temp_pred_interp = self.interpolate_output(
                x, y, [temp_true, temp_pred], extent
            )

            # Create figure
            f = plt.figure(figsize=(16, 6), dpi=100)
            plt.suptitle(f"Magma Chamber at {t*endTime:.1f} years", fontsize=16)
            
            # True temperature
            plt.subplot(1, 3, 1)
            plt.title("True Temperature")
            im1 = plt.imshow(temp_true_interp.T, origin="lower", extent=extent, 
                           cmap='hot', vmin=min_true_temp, vmax=max_true_temp)
            plt.colorbar(im1, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Predicted temperature
            plt.subplot(1, 3, 2)
            plt.title("Predicted Temperature")
            im2 = plt.imshow(temp_pred_interp.T, origin="lower", extent=extent, 
                           cmap='hot', vmin=min_pred_temp, vmax=max_pred_temp)
            plt.colorbar(im2, label="Temperature (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            
            # Difference plot
            plt.subplot(1, 3, 3)
            plt.title("Difference (Pred - True)")
            diff = temp_pred_interp - temp_true_interp
            im3 = plt.imshow(diff.T, origin="lower", extent=extent, 
                           cmap='RdBu_r', vmin=-50, vmax=50)
            plt.colorbar(im3, label="Temperature Difference (°C)")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")

            plt.tight_layout()
            figures.append((f, f"temp_t{t*endTime:.1f} years"))
            
        return figures
    
    @staticmethod 
    def interpolate_output(x, y, us, extent):
        """Interpolates irregular points onto a mesh"""
        xyi = np.meshgrid(
            np.linspace(extent[0], extent[1], 100), 
            np.linspace(extent[2], extent[3], 100), 
            indexing="ij"
        )
        us = [scipy.interpolate.griddata((x, y), u.ravel(), tuple(xyi), 
                                       method='nearest', fill_value=np.nan) for u in us]
        return us