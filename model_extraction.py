import json
import os
import keras
from keras.models import model_from_json


def save_model(model, name='model', path='models/'):
    model_name = name + '.h5'
    save_dir = os.path.join(os.getcwd(), path)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    model_json = model.to_json()
    with open(path + name + '.json', "w") as json_file:
        json_file.write(model_json)


def load_model(name='model', path='models/'):
    json_file = open(path+name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+name+'.h5')
    print("Loaded model from disk")
    return loaded_model
