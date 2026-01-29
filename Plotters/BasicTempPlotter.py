import matplotlib.pyplot as plt
import numpy as np
from physicsnemo.sym.utils.io.plotter import ValidatorPlotter
import scipy

class ChamberPlotter(ValidatorPlotter):
    def __call__(self, invar, true_outvar, pred_outvar):
        chamber_width, chamber_height = 20000, 6000  # m
        x_min_km, x_max_km = 0.0, chamber_width / 1000.0
        y_min_km, y_max_km = 0.0, chamber_height / 1000.0  # depth 0–6 km

        timeScalingFactor = 1000000.0
        tempScalingFactor = 1000.0

        times = np.unique(invar["time"][:, 0])
        figures = []

        min_true_temp = max(true_outvar["Temperature"].min(), -0.00001)
        min_pred_temp = min(pred_outvar["Temperature"].min(), -0.00001)

        max_true_temp = max(true_outvar["Temperature"].max() * tempScalingFactor, 901)
        max_pred_temp = min(pred_outvar["Temperature"].max() * tempScalingFactor, 901)

        for t in times:
            time_mask = (invar["time"][:, 0] == t)
            x = invar["x"][time_mask, 0]  # normalized 0–1
            y = invar["y"][time_mask, 0]  # normalized 0–1

            if len(x) == 0:
                continue

            # Interpolate in normalized space
            norm_extent = (0.0, 1.0, 0.0, 1.0)

            temp_true = true_outvar["Temperature"][time_mask] * tempScalingFactor
            temp_pred = pred_outvar["Temperature"][time_mask] * tempScalingFactor

            temp_true_interp, temp_pred_interp = self.interpolate_output(
                x, y, [temp_true, temp_pred], norm_extent
            )

            # Figure: two 20×6 km frames stacked
            f = plt.figure(figsize=(14, 10), dpi=300)
            plt.suptitle(f"Magma Chamber at {t * timeScalingFactor:.1f} years", fontsize=16)

            # Plot extent in km, with depth increasing downward
            plot_extent = (x_min_km, x_max_km, y_max_km, y_min_km)

            '''
            # True temperature
            plt.subplot(2, 1, 1)
            plt.title("True Temperature")
            im1 = plt.imshow(
                temp_true_interp,
                origin="upper",
                extent=plot_extent,
                cmap="jet",
                vmin=min_true_temp,
                vmax=max_true_temp,
                aspect="auto",
                interpolation="bicubic"
            )
            plt.colorbar(im1, label="Temperature (°C)")
            plt.xlabel("Distance (km)")
            plt.ylabel("Depth (km)")
            '''

            # Predicted temperature
            plt.subplot(2, 1, 1)
            plt.title("Predicted Temperature")
            im2 = plt.imshow(
                temp_pred_interp,
                origin="upper",
                extent=plot_extent,
                cmap="jet",
                vmin=min_pred_temp,
                vmax=max_pred_temp,
                aspect="auto",
                interpolation="bicubic"
            )
            plt.colorbar(im2, label="Temperature (°C)")
            plt.xlabel("Distance (km)")
            plt.ylabel("Depth (km)")

            plt.tight_layout()
            figures.append((f, f"temp_t{t * timeScalingFactor:.1f} years"))

        return figures
    
    @staticmethod
    def interpolate_output(x, y, us, extent):
        """Interpolates irregular points onto a (y, x) mesh so imshow displays correctly"""

        # extent is (xmin, xmax, ymin, ymax) in normalized space
        yi = np.linspace(extent[2], extent[3], 100)  # 0 → 1
        xi = np.linspace(extent[0], extent[1], 100)  # 0 → 1

        Y, X = np.meshgrid(yi, xi, indexing="ij")  # shape: [ny, nx]

        out = [
            scipy.interpolate.griddata(
                (x, y),
                u.ravel(),
                (X, Y),
                method="nearest",
                fill_value=np.nan,
            )
            for u in us
        ]

        return out