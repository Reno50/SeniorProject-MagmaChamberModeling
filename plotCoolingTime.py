import os
import csv
import matplotlib.pyplot as plt

with open('outputs/NewEquationsModel/coolingTimes.csv') as coolingTimesFile:
    times = [float(value) for value in coolingTimesFile.read().split(',') if value.strip()] # There should be 100 entries in here
    plt.plot([x * (50000/100) for x in range(len(times))], times) # x, y - step, cooling time predicted at said step
    plt.xlabel('Step')
    plt.ylabel('Cooling time predicted at step')
    plt.title('Predicted cooling time at each point in training')
    plt.savefig('cooling_time_plot.png', dpi=300, bbox_inches='tight')
    print('Plot saved to outputs/NewEquationsModel/cooling_time_plot.png')