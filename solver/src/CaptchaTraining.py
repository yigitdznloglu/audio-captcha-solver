import numpy as np
import h5py
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from scipy import stats
import scipy
import pickle
from rastaplp import rastaplp
from joblib import load as joblib_load
from joblib import dump as joblib_dump


"""
DEFINITIONS
    Ceps (Cepstral coefficients): Coefficients representing the audio signal in the cepstral domain,
        derived from the signal's spectral properties. They capture frequency and time characteristics
        of phonetic units.
    HDF5 (Hierarchical Data Format version 5): A file format for storing and organizing large amounts
        of data.
    Label: Identifiers for each audio sample, such as correct digits or 'noise'.
    PCA (Principal Component Analysis) Transformation: A statistical technique for reducing data
        dimensionality, transforming correlated variables into uncorrelated principal components.
    Spec (Spectrogram): A visual representation of a sound signal's frequency spectrum over time.
    SVC (Support Vector Classifier): A machine learning model that classifies data points into
        different categories using hyperplanes in high-dimensional space.
        TELL ME LIKE IM 5: Imagine you have a bunch of colored balls scattered on a table, and you
        want to separate them by color using a flat piece of cardboard. In this case, the cardboard
        is like the "hyperplane" in SVC. SVC, which is a smart tool in machine learning, helps find
        the best position and angle to place the cardboard so that balls of one color are on one
        side and the others are on the opposite side. This way, it helps us tell which balls belong
        to which color group, even if there are a lot of them and they are close together!
"""

class CaptchaTraining:
    """
    The CaptchaAnalysis class is designed to process and analyze audio CAPTCHA data. It provides
    functionality to load data from HDF5 files, preprocess and transform the data for machine 
    learning, and to train and test models for CAPTCHA recognition.

    Attributes:
        X (dict): A dictionary to hold the structure of the HDF5 file groups.
        iii (int): An index counter used when traversing the HDF5 file structure.
    """

    def __init__(self):
        """
        Constructor for the CaptchaAnalysis class. It initializes the class with default values.

        Attributes:
        X (dict): A dictionary to hold the structure of the HDF5 file groups.
        iii (int): An index counter used when traversing the HDF5 file structure.
        """
        self.X = {}  # Initialize a dictionary to store HDF5 file structure
        self.iii = 0  # Initialize a counter to keep track of the HDF5 group index


    def take_hdf5_item_structure(self, g, offset='    '):
        """
        Recursively traverses the structure of an HDF5 file/group, records the names of groups
        within the file structure, and prints the keys. It updates the instance variable `X` with the 
        names of the groups and datasets.

        Args:
        g: The HDF5 group or file to traverse.
        offset (str): A string used for formatting the print output to represent the hierarchy.
        """
        # If the object is a group, store its name in the 'X' dictionary under the current index ('iii')
        if isinstance(g, h5py.Group):
            self.X[self.iii] = g.name
            print(f"{offset}Group: {g.name}")  # Print the group name
            self.iii += 1

        # If the object is a file or a group, iterate through its items
        if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
            for key, val in dict(g).items():
                subg = val
                if isinstance(subg, h5py.Dataset):
                    print(f"{offset}Dataset: {subg.name}")  # Print the dataset name
                # Recursively call this function for each sub-group
                self.take_hdf5_item_structure(subg, offset + '    ')

    def loaddata(self, matfile):
        # Open the HDF5 file
        file = h5py.File(matfile, 'r')  
        self.take_hdf5_item_structure(file)

        # Initialize dictionaries for storing data
        Ceps = {}
        Spec = {}
        Label = {}

        # Read and store Ceps, Spec, and Labels from the file
        for i in range(2, len(self.X)):
            # Check if the key is a reference to a nested group
            if self.X[i].startswith('/#refs#/'):
                key_prefix = self.X[i]
            else:
                # If not a nested group, handle as a direct dataset
                # Add appropriate logic here if needed
                key_prefix = '/#refs#/' + self.X[i]

            Ceps[i-2] = np.array(file[key_prefix + '/ceps'])
            Spec[i-2] = np.array(file[key_prefix + '/spec'])
            Label[i-2] = file[key_prefix + '/label'][0]

        # Close the file
        file.close()
        print(len(Ceps), ' number of data read from given dataset')

        # Prepare arrays for training
        m, n = Ceps[0].shape
        self.Ftrain = np.zeros((len(Ceps), m*n))
        self.FtrLabel = np.zeros(len(Ceps))

        # Flatten and store Ceps and Labels
        for i in range(0, len(Ceps)):
            self.Ftrain[i, :] = Ceps[i].flatten()
            self.FtrLabel[i] = Label[i] + 1

    
    def train(self, cost=1, pcavar=0.95):
        """
        Trains a model for audio CAPTCHA recognition. This involves several steps:
        - Applying PCA for dimensionality reduction.
        - Training an SVM (SVC) classifier.
        - Evaluating the classifier and saving the trained model.

        Args:
        cost (float): The regularization parameter for the SVM classifier.
        pcavar (float): Threshold for the cumulative variance in PCA.

        Returns:
        A matrix with success rates for each class in the training data.
        """
        self.loaddata(r"../dataset/train/train_features.mat")
        # Since the data is already loaded and processed in loaddata, we use self.Ftrain and self.FtrLabel directly
        train = self.Ftrain
        trLabel = self.FtrLabel

        # Standardize the data (to have a mean of 0 and a standard deviation of 1)
        zstrain = stats.zscore(train)
        
        # Initialize a PCA transformation
        if not hasattr(self, 'pca'):
            self.pca = PCA(n_components=200)
            self.pca.fit(zstrain)

        # Calculate the cumulative sum of explained variance ratio
        csv = np.cumsum(self.pca.explained_variance_ratio_)
        
        # Determine the number of principal components to keep, based on the pcavar threshold
        self.el = np.where(csv > pcavar)[0][0]

        # Apply PCA to the original data
        train_pc = self.pca.transform(train)[:, 0:self.el+1]

        # Initialize an SVC with a cost parameter and assign it to self.clf (classifier)
        self.clf = SVC(C=1)
        self.clf.fit(train_pc, trLabel)

        # Use the trained model to predict labels for the training data
        w = self.clf.predict(train_pc) 
        results = np.zeros((11, 2))
        for i in range(0, 11):
            results[i, 0] = np.sum(w[trLabel == i+1] == i+1)
            results[i, 1] = np.sum(trLabel == i+1)
        
        # Save the model, SVC classifier, and other parameters using joblib
        model_data = (self.pca, self.el, self.clf)
        filen = "model_" + str(cost) + "_" + str(pcavar) + ".joblib"
        joblib_dump(model_data, filen)

        # Return results, which contains the success rates for each class in the training data
        return results
    

    def loadtrainparam(self, traineddatafile=""):
        """
        Loads trained model parameters using joblib.

        Args:
        traineddatafile (str, optional): The path to the file containing the trained model parameters.
        """
        if traineddatafile == "":
            with open("last_train_filename", "r") as file:
                traineddatafile = file.read().strip()

        # Load the PCA model, number of elements (el), and the classifier (clf) using joblib
        loaded_data = joblib_load(traineddatafile)

        # Check if loaded_data is a tuple with 3 elements
        if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
            self.pca, self.el, self.clf = loaded_data
        else:
            raise ValueError("The loaded model data does not contain the expected components.") 
    
    def running_mean(self,x, N):
        cumsum = np.cumsum(x) #(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / N 

    def test(self, testfile):
        """
        Tests a given audio file against the trained PCA and SVM model.

        Args:
        testfile (str): The path to the audio file to be tested.

        The method processes the audio file, extracts features, applies PCA transformation,
        and then uses the trained SVM classifier to predict the digits in the audio. The
        final result is a string of predicted digits.

        Returns:
        str: The string of predicted digits from the audio file.
        """
        # Calculate the expected number of digits in the test file name
        digit_number = len(testfile) - 4

        # Read the audio file and preprocess the signal
        fs, ff = scipy.io.wavfile.read(testfile)
        length = len(ff)
        f = ff[0 : int(np.floor(length / 2))] / 32768.0
        energy_f = np.abs(f) * f * f
        y = self.running_mean(energy_f, 100)
        mean_locs = (y > 0.0003)
        zero_locs = (y < 0.00001)
        flag = 0

        # Initialize variables for locating and processing digit sounds
        location = {}
        cnt = 0
        index = -1
        startpoint = 0
        endpoint = 0
        segment = {}
        ceps = {}
        spec = {}
        svmLabel = {}
        digit_count = 0
        result = ""

        # Identify segments in the audio that might contain digits
        for i in range(len(y)):
            # Logic to identify start and end points of potential digit segments
            if flag == 0 and mean_locs[i]:
                location[cnt] = i
                cnt += 1
                flag = 1
            elif flag == 1 and not mean_locs[i]:
                location[cnt] = i - 1
                cnt += 1
                flag = 0

        # Process each identified segment
        for i in range(1, 1 + int(np.floor(len(location) / 2))):
            if location[2 * i - 1] - location[2 * i - 2] > 200:
                # Determine the starting point of the segment
                for j in range(int(location[2 * i - 2]), -1, -1):
                    if zero_locs[j]:
                        startpoint = j
                        break
                if startpoint < np.floor((location[2 * i - 2] + location[2 * i - 1]) / 2) - 1750:
                    startpoint = int(np.floor((location[2 * i - 2] + location[2 * i - 1]) / 2) - 1750)

                # Extract, process, and classify the segment if it's a potential digit
                if startpoint > endpoint and startpoint + 3500 < len(f):
                    index += 1
                    segment[index] = f[startpoint : startpoint + 3501]
                    ceps[index], spec[index], _, _, _, _ = rastaplp(segment[index], fs, 0, 12)
                    endpoint = startpoint

                    # Transform the features and classify using the trained SVM model
                    m, n = ceps[index].shape
                    testd = np.zeros((1, m * n))
                    testd[0, :] = ceps[index].flatten()
                    test_pc_pca = self.pca.transform(testd)
                    test_pc = test_pc_pca[:, 0 : self.el + 1]
                    svmLabel[index] = self.clf.predict(test_pc)

                    # Append the result if a digit is recognized
                    if 1 <= svmLabel[index] <= 10:
                        endpoint = startpoint + 3500
                        digit_count += 1
                        result += str(int(svmLabel[index][0]) - 1)
                    else:
                        endpoint = startpoint

                # Stop processing if the expected number of digits is reached
                if digit_count == digit_number:
                    break

        return result
    
if __name__ == "__main__":    
    ca=CaptchaTraining()     
    ca.loadtrainparam('model_100_0.95.joblib')

    result=ca.test('..dataset/test/04648.wav')
    print (result)