import os
import csv
import matplotlib.pyplot as plt

with open('outputs/NewEquationsModel/lossFile.csv') as lossFile:
    losses = [float(value) for value in lossFile.read().split(',') if value.strip()] # There should be 50k entries in here
    plt.plot([x for x in range(len(losses))], losses) # x, y - steps, loss at a step
    plt.xlabel('Step')
    plt.ylabel('Loss at step')
    plt.title('Loss per step')
    plt.savefig('outputs/NewEquationsModel/loss_plot.png', dpi=300, bbox_inches='tight')
    print('Plot saved to outputs/NewEquationsModel/loss_plot.png')