import keras
from keras.models import load_model
import pickle
import os
import numpy as np

model_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'keras_cifar100_45%_model.h5'
model_name = 'trained_cifar100_trained_model_2017-10-30_____15-29-00.h5'
model_path = os.path.join(model_dir, model_name)
model = load_model(model_path)

with open('./test_data', 'rb') as f:
    test_data = pickle.load(f)
    # test_label = pickle.load(f)


# # one_hot_test_predictions = model.predict(test_data.reshape(10000,32,32,3))
# # one_hot_test_predictions = model.predict(test_data.reshape(10000,3,32,32))
x_test = test_data.reshape(10000,3,32,32).transpose(0,2,3,1)
one_hot_test_predictions = model.predict(x_test)
not_hot_test_predictions = np.round(np.argmax(one_hot_test_predictions, axis = 1))

# predictions_dir = os.path.join(os.getcwd(), 'saved_predictions')
# predictions_name = model_name[:-3]
# predictions_path = os.path.join(predictions_dir, predictions_name)

# if not os.path.isdir(predictions_dir):
#     os.makedirs(predictions_dir)

# # np.savetxt(predictions_path, not_hot_test_predictions, delimiter=",", format='%d')
# not_hot_test_predictions.tofile(predictions_path,sep='\n',format='%d')





# from keras.datasets import cifar100
# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# num_classes = 100
# y_test = keras.utils.to_categorical(y_test, num_classes)

# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])






from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_test = np.squeeze(y_test)


true_labels = y_test
pred_labels = not_hot_test_predictions

assert(len(true_labels) == len(pred_labels))
print('Test accuracy:', np.sum(true_labels == pred_labels) / len(true_labels))
