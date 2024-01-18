from CaptchaTraining import CaptchaTraining
import os
import sys
import numpy as np


"""
DEFINITIONS

Dynamic Time Warping (DTW):  (DTW) is an algorithm used for measuring similarity between two
    temporal sequences which may vary in speed. In the context of this code, it's used to compare two
    audio sequences (like those in CAPTCHA tests) by aligning them in time. DTW calculates the optimal
    match between these sequences by stretching or compressing them along the time axis. This method
    is particularly useful in audio and speech processing as it can robustly handle differences in
    speaking speed and timing variations between different audio samples.
"""

# Function to calculate the distance between two points in the sequence
def dist(A, B, s):
    """Calculates the distance between two points in the sequence.
    
    Args:
    A, B: Points in the sequence to compare.
    s: Flag to indicate if the comparison is at the end of the sequence.
    
    Returns:
    int: The distance between the points.
    """
    if A == B:
        return 0
    return 1  # Returns 1 for non-matching points

# Function to perform Dynamic Time Warping Analysis
def dynamicTimeWarpAnalysis(seqA, seqB, nap):
    """Performs Dynamic Time Warping (DTW) (See DEFINITIONS for Dynamic Time Warping) on two
    sequences.
    
    Args:
    seqA, seqB: The two sequences to be compared using DTW.
    nap: The neighborhood around the current point to consider in the comparison.
    
    Returns:
    float: The DTW cost of aligning the two sequences.
    """
    numRows, numCols = len(seqA), len(seqB)
    cost = [[np.inf for _ in range(numCols)] for _ in range(numRows)]
    cost[0][0] = dist(seqA[0], seqB[0], False)

    # Initializes the first row and column of the cost matrix
    for i in range(1, numRows):
        cost[i][0] = cost[i-1][0] + dist(seqA[i], seqB[0], False)
    for j in range(1, numCols):
        cost[0][j] = 0

    # Computes the cost for aligning the two sequences
    for i in range(1, numRows):
        jj = i * numCols / numRows
        bas = max(1, int(jj) - nap)
        son = min(numCols, int(jj) + nap)
        for j in range(bas, son):
            choices = [cost[i-1][j], cost[i][j-1], cost[i-1][j-1]]
            cost[i][j] = min(choices) + dist(seqA[i], seqB[j], j == numCols-1)

    return cost[-1][-1]  # Returns the final alignment cost

# Prints usage instructions
print('Using instruction......')
print('python captcha_test traineddatafile [filename1 | --all] <filename2> <filename3>, .....')
print('---------------------')

# Default file paths for the trained data file and the parameter
traineddatafile= "model_100_0.95.joblib"
param = "--all"

# Overrides default values with command line arguments if provided
if len(sys.argv) > 1:
    traineddatafile = str(sys.argv[1])    
if len(sys.argv) > 2:
    param = str(sys.argv[2])

# Initializes the CaptchaTraining object and loads trained parameters
ca = CaptchaTraining() 
ca.loadtrainparam(traineddatafile)

print('Using trained data file: ', traineddatafile)

# Initializes counters for precision calculation
d = 0  # Count of correctly identified files
t = 0  # Total count of files processed
dd = 0 # Cumulative difference in DTW cost
tt = 0 # Total length of file labels

# Tests either all files in a directory or specific files provided as arguments
if param == "--all":
    # Iterates over all .wav files in the test dataset directory
    for file in os.listdir('../dataset/test'):
        if file.endswith(".wav"):
            result = ca.test('../dataset/test/' + str(file))
            cost = dynamicTimeWarpAnalysis(result, file[0:-4], 4)
            tt += len(file[0:-4])
            dd += len(file[0:-4]) - cost

            if result == file[0:-4]:
                d += 1
            t += 1
            print(str(file), '\t------->', result, '\t precision: ', d*1.0/t, '\t digit precision: ', dd*1.0/tt)
else:
    # Tests only the files provided as arguments
    for i in range(2, len(sys.argv)):
        result = ca.test(str(sys.argv[i]))
        fn = sys.argv[i]
        if result == fn[0:-4]:
            d += 1
        t += 1
        print(str(sys.argv[i]), '\t------->', result, '\t precision: ', d*1.0/t)

# Prints the final precision statistics
print('General precision:', d*1.0/t, '\t digit precision:', dd*1.0/tt)
