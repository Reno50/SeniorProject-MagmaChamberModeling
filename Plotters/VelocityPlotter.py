import matplotlib.pyplot as plt
import numpy as np
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import scipy

class VelocityPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        chamber_width, chamber_height = 20000, 6000  # m
        x_min_km, x_max_km = 0.0, chamber_width / 1000.0
        y_min_km, y_max_km = 0.0, chamber_height / 1000.0  # depth 0–6 km

        timeScalingFactor = 1000000.0
        tempScalingFactor = 1000.0

        times = np.unique(invar["time"][:, 0])
        figures = []

        min_pred_temp = pred_outvar["Temperature"].min() * tempScalingFactor
        max_pred_temp = pred_outvar["Temperature"].max() * tempScalingFactor

        for t in times:
            time_mask = (invar["time"][:, 0] == t)
            x = invar["x"][time_mask, 0]
            y = invar["y"][time_mask, 0]

            if len(x) == 0:
                continue

            norm_extent = (0.0, 1.0, 0.0, 1.0)

            temp_pred = pred_outvar["Temperature"][time_mask] * tempScalingFactor
            vel_x = pred_outvar["XVelocity"][time_mask]
            vel_y = pred_outvar["YVelocity"][time_mask]

            # Interpolate temperature and velocities onto grid
            temp_pred_interp, vel_x_interp, vel_y_interp = self.interpolate_output(
                x, y, [temp_pred, vel_x, vel_y], norm_extent
            )

            # Create figure
            f = plt.figure(figsize=(16, 8), dpi=300)
            plt.suptitle(f"Temperature and Velocity at {t * timeScalingFactor / 1000:.0f} kyr", fontsize=16)

            # Plot extent in km, with depth increasing downward
            plot_extent = (x_min_km, x_max_km, y_max_km, y_min_km)

            # Predicted temperature with velocity vectors
            ax = plt.subplot(1, 1, 1)
            
            # Temperature field
            im = plt.imshow(
                temp_pred_interp,
                origin="upper",
                extent=plot_extent,
                cmap="jet",  # Similar to the reference image
                vmin=min(min_pred_temp, 50),
                vmax=min(max_pred_temp, 900),
                aspect="auto",
                interpolation="bicubic"
            )
            
            # Create meshgrid for velocity vectors
            # Use fewer points for cleaner visualization
            n_arrows_x = 25
            n_arrows_y = 15
            yi_arrow = np.linspace(0, 1, n_arrows_y)
            xi_arrow = np.linspace(0, 1, n_arrows_x)
            Y_arrow, X_arrow = np.meshgrid(yi_arrow, xi_arrow, indexing="ij")
            
            # Interpolate velocities at arrow positions
            vel_x_arrows = scipy.interpolate.griddata(
                (x, y), vel_x.ravel(), (X_arrow, Y_arrow),
                method="linear", fill_value=0.0
            )
            vel_y_arrows = scipy.interpolate.griddata(
                (x, y), vel_y.ravel(), (X_arrow, Y_arrow),
                method="linear", fill_value=0.0
            )
            
            # Convert arrow positions to km for plotting
            X_arrow_km = X_arrow * (x_max_km - x_min_km) + x_min_km
            Y_arrow_km = Y_arrow * (y_max_km - y_min_km) + y_min_km
            
            # Plot velocity vectors
            # Note: Y increases downward in the plot, so we may need to flip vel_y
            quiver = plt.quiver(
                X_arrow_km, Y_arrow_km,
                vel_x_arrows, -vel_y_arrows,  # Negative Y for downward-positive depth
                color='black',
                alpha=0.7,
                scale=None,
                scale_units='xy',
                width=0.003,
                headwidth=3,
                headlength=4
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, label="Temperature (°C)", pad=0.02)
            
            plt.xlabel("Distance (km)", fontsize=12)
            plt.ylabel("Depth (km)", fontsize=12)
            
            plt.tight_layout()
            figures.append((f, f"velocity_t{t * timeScalingFactor / 1000:.0f}kyr"))
        return figures
    
    @staticmethod
    def interpolate_output(x, y, us, extent):
        """Interpolates irregular points onto a (y, x) mesh so imshow displays correctly"""

        # extent is (xmin, xmax, ymin, ymax) in normalized space
        yi = np.linspace(extent[2], extent[3], 150)  # 0 → 1
        xi = np.linspace(extent[0], extent[1], 150)  # 0 → 1

        Y, X = np.meshgrid(yi, xi, indexing="ij")  # shape: [ny, nx]

        out = [
            scipy.interpolate.griddata(
                (x, y),
                u.ravel(),
                (X, Y),
                method="linear",
                fill_value=np.nan,
            )
            for u in us
        ]

        return out
