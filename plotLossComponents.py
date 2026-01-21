# Generates the figure with each individual loss component in the same graph
import csv
import matplotlib.pyplot as plt
import numpy as np

try: 
    with open('lossComponents.csv') as comp_file:
        reader = csv.DictReader(comp_file)
        data = list(reader)
except FileNotFoundError:
    print("Could not find lossComponents.csv!")
    exit(1)
if not data:
    print("No data found in lossComponents.csv!")
    exit(1)

# Extract column names (excluding step and total_loss)
component_names = [col for col in data[0].keys() if col not in ['step', 'total_loss']]

# Prepare data for plotting
steps = [int(row['step']) for row in data]
total_loss = [float(row['total_loss']) for row in data]

components = {}
for name in component_names:
    components[name] = [float(row[name]) for row in data]

# Create the plot
plt.figure(figsize=(14, 8))

# Plot total loss with thicker line
plt.plot(steps, total_loss, label='Total Loss', linewidth=2.5, color='black', alpha=0.8)

# Plot individual components with different colors
colors = plt.cm.tab20(np.linspace(0, 1, len(component_names)))
for i, name in enumerate(sorted(component_names)):
    plt.plot(steps, components[name], label=name, linewidth=1.5, alpha=0.7, color=colors[i])

plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Loss Components Over Training', fontsize=14)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('outputs/NewEquationsModel/loss_components_plot.png', dpi=300, bbox_inches='tight')
print(f'Plot saved to outputs/NewEquationsModel/loss_components_plot.png')
print(f'Total components plotted: {len(component_names) + 1}')  # +1 for total loss
