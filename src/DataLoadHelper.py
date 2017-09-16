import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class DataLoadHelper:
    """Reads data generated from the simulator and return the training data as numpy arrays"""
    csv_file_path=None
    train_samples = None
    validation_samples = None

    def __init__(self, csv_file_name):
        self.csv_file_path = csv_file_name
        self.__get_samples__()

    def sample_training_size(self):
        """
        return train and validation sample data size
        """
        return len(self.train_samples)

    def sample_validation_size(self):
        """
        return validation sample data size
        """
        return len(self.validation_samples)

    def __read_csv__(self):
        """Read the csv from the given file name and return all the lines"""
        lines = []
        with open(self.csv_file_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        return lines

    def __get_samples__(self):
        """
        get samples from csv files and split them in test and train data
        samples are the csv rows (not the images data
        """
        self.train_samples, self.validation_samples = train_test_split(self.__read_csv__(), test_size=0.2)

    def load_all_data(self):
        """loads all the images from the path in the csv file as X_train and steering angle as y_train
        if the data is bigger then we can't fit all the data in the memory at once. So be careful in using this method
        switch to generator if you face issues"""
        images = []
        measurements = []
        lines = self.__read_csv__(self.csv_file_path)
        for line in lines:
            file_name = line[0][0].split('/')[-1]
            imageFolder = line[0][0].split('/')[-2]
            csv_split_path = self.csv_file_path.split('/')
            # path format '../data/IMG/filename'
            center_image_path = csv_split_path[0] + '/' + csv_split_path[1] + '/' + imageFolder + '/' + file_name
            image = cv2.imread(center_image_path)
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)

        X_train = np.array(images, dtype=float)
        y_train = np.array(measurements, dtype=float)

        return X_train, y_train

    def __generator__(self, samples, batch_size = 32):
        """
        loads the data of 'batch size' by randomly selecting rows from the given 'sample' rows of csv file
        only reads center images and flips them vertically
        :param samples:
        :param batch_size:
        :return: yield of length 'batch size' with the  data
        """
        num_samples = len(samples)

        while 1:
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                steer_angles = []
                for batch_sample in batch_samples:
                    try:
                        file_name = batch_sample[0].split('/')[-1]
                        imageFolder = batch_sample[0].split('/')[-2]
                        csv_split_path = self.csv_file_path.split('/')
                        #path format '../data/IMG/filename'
                        center_image_path = csv_split_path[0] + '/' + csv_split_path[1] + '/' + imageFolder + '/' + file_name
                        center_image = cv2.imread(center_image_path)
                        center_image_steer_angle = float(batch_sample[3])
                        images.append(center_image)
                        steer_angles.append(center_image_steer_angle)
                    except:
                        print("Got exception in ", batch_sample[3])
                X_train = np.array(images)
                y_train = np.array(steer_angles)
                yield shuffle(X_train, y_train)

    def load_train_data_from_generator(self, bath_size = 32):
        """
        loads training data in batches of size 'batch_size'
        this functions uses python generators
        :param bath_size: length of required data
        :return: python generator of the required data
        """
        return self.__generator__(self.train_samples, bath_size)

    def load_validation_data_from_generator(self, bath_size = 32):
        """
        loads validation data in batches of size 'batch_size'
        this functions uses python generators
        :param bath_size: length of required data
        :return: python generator of the required data
        """
        return self.__generator__(self.validation_samples, bath_size)


