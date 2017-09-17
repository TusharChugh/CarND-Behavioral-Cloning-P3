from DataLoadHelper import *
from model import model
from keras import backend as K
import matplotlib.pyplot as plt


#Load data collected from the simulator
csv_data_path = "../data/driving_log.csv"
output_model_file_path= '../results/models/model.h5'

simulator_data = DataLoadHelper(csv_data_path)
train_generator = simulator_data.load_train_data_from_generator()
validation_generator = simulator_data.load_validation_data_from_generator()

#Clear any previous keras sessions
K.clear_session()

#Create an object to the model, compile, fit and save it to .h5 file
model = model(input_size = (160,320, 3))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=simulator_data.sample_training_size(), \
                    validation_data=validation_generator, nb_val_samples=simulator_data.sample_validation_size(),
                    nb_epoch=3, verbose=1)


model.save(output_model_file_path)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

