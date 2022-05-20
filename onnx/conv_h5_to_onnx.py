from tensorflow import keras
import onnx
import keras2onnx
import sys

path_to_tf2_models = "../static_res/tf2Models/"
path_to_onnx_models = "../static_res/onnxModels/"

model = keras.models.load_model(path_to_tf2_models + sys.argv[1])
onnx_model = keras2onnx.convert_keras(model, model.name)

file = open(path_to_onnx_models + sys.argv[2], "wb")
file.write(onnx_model.SerializeToString())
file.close()