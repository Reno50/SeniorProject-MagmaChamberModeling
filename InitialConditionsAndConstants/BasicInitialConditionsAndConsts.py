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
        elif (x_flat[i] > (5000/chamber_width)) or (y_flat[i] > (2400/chamber_height)): # Smooth the boundary
            # The x value - 0 meaning at 5000, 1 meaning 6000
            # The y value - same thing ish
            x = ((x_flat[i] - (5000/chamber_width)) * chamber_width) / 1000
            y = ((y_flat[i] - (2400/chamber_height)) * chamber_height) / 600
            num = max(x, y)
            returnVals[i] = num * 170.0 - 25.0 * (y_flat[i] / (1000.0/chamber_height)) + (1-num) * 900
            # Basically average the two
        else:
            returnVals[i] = 900
    
    return returnVals