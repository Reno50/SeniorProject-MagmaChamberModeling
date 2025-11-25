import numpy as np

# Functions on the condition that the top left point is 0, 0  and bottm left is 1, 1

tempScalingFactor = 1000.0 # 1000 degrees is 1.0 in the network

def generate_initial_temps(x, y): # A list of temperatures at each sample point
    '''
    Given two nparrays, return the initial temp - pretty simple, going off the diagram on page 171 of the paper
    '''
    chamber_width, chamber_height = 20000, 6000 # Same for chamber size - normalize to 0 - 1
    # Now they are one dimensional arrays
    x_flat = np.array(x).flatten() # a number of Xs
    y_flat = np.array(y).flatten() # an identical number of Ys

    returnVals: list[float] = [0 for i in range(len(x_flat))] # an identical number of 0s for each point

    for i in range(len(returnVals)):
        # Define transition zones
        # Cold region boundary: x > 6000 or y < 3000
        # Hot region boundary: x < 5000 and y > 3600
        
        x_val = x_flat[i] * chamber_width
        y_val = y_flat[i] * chamber_height
        
        # Calculate interpolation weight w (0 = Hot, 1 = Cold)
        
        # X component of weight
        if x_val <= 5000:
            wx = 0.0
        elif x_val >= 6000:
            wx = 1.0
        else:
            wx = (x_val - 5000.0) / 1000.0
            
        # Y component of weight
        if y_val >= 3600:
            wy = 0.0
        elif y_val <= 3000:
            wy = 1.0
        else:
            wy = (3600.0 - y_val) / 600.0
            
        w = max(wx, wy)
        
        cold_temp = (20.0 + 25.0 * (y_flat[i] / (1000.0/chamber_height)))
        hot_temp = 900.0
        
        returnVals[i] = (w * cold_temp + (1.0 - w) * hot_temp) / tempScalingFactor
    
    return np.array(returnVals).reshape(-1, 1)