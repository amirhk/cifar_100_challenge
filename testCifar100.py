from keras.models import load_model
import pickle
import os
import numpy as np

model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar100_45%_model.h5'
model_path = os.path.join(model_dir, model_name)
model = load_model(model_path)

with open('./test_data', 'rb') as f:
    test_data = pickle.load(f)
    # test_label = pickle.load(f)


one_hot_test_predictions = model.predict(test_data.reshape(10000,32,32,3))
not_hot_test_predictions = np.round(np.argmax(one_hot_test_predictions, axis = 1))

predictions_dir = os.path.join(os.getcwd(), 'saved_predictions')
predictions_name = 'keras_cifar100_45%_pred.csv'
predictions_path = os.path.join(predictions_dir, predictions_name)

if not os.path.isdir(predictions_dir):
    os.makedirs(predictions_dir)

# np.savetxt(predictions_path, not_hot_test_predictions, delimiter=",", format='%d')
not_hot_test_predictions.tofile(predictions_path,sep='\n',format='%d')
