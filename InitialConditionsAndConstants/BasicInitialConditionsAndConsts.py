import numpy as np

def generate_initial_temps(x, y) -> list[int]: # A list of temperatures at each sample point
    '''
    Given two nparrays, return the initial temp - pretty simple, going off the diagram on page 171 of the paper
    '''
    chamber_width, chamber_height = 20000, 6000 # Same for chamber size - normalize to 0 - 1
    # Now they are one dimensional arrays
    x_flat = np.array(x).flatten() # a number of Xs
    y_flat = np.array(y).flatten() # an identical number of Ys

    returnVals = [0 for i in range(len(x_flat))] # an identical number of 0s for each point

    for i in range(len(returnVals)):
        if (x_flat[i] > (6000/chamber_width)) or (y_flat[i] > (3000/chamber_height)):
            returnVals[i] = 170.0 - 25.0 * (y_flat[i] / (1000.0/chamber_height))
        else:
            returnVals[i] = 900 - (700 * y_flat[i]) - (700 * x_flat[i])
    
    return returnVals