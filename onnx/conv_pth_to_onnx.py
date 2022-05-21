import torch as th
import sys

if __name__=="__main__": 
  path_to_pth_models = "../static_res/pthModels/"
  path_to_onnx_models = "../static_res/onnxModels/"
  path_to_pth_src = "../pytorch/"
  arg = sys.argv[1]
  path_to_pth_src_model = "../pytorch/" + arg

  sys.path.insert(0, path_to_pth_src_model)

  from NN_def import NN
  from NN_def import get_input_size

  model = NN()
  model.load_state_dict(th.load(path_to_pth_models + arg + "_model.pth"))

  model.eval()
  input_size = list(get_input_size(path_to_pth_src_model))
  dummy_in = th.randn(input_size)#, requires_grad=True)

  th.onnx._export(model, dummy_in, path_to_onnx_models + arg + "_model.onnx", export_params=True)

  #th.onnx.export(
  #  model,
  #  dummy_in,
  #  path_to_onnx_models + arg + "_model.onnx",
  #  export_params=True,
  #  opset_version=10, # onnx version
  #  #do_constant_folding=True, # ???
  #  input_names=["in"],
  #  output_names=["out"]
  #  #dynamic_axes={"in" : {0 : "batch_size"}, "out" : {0 : "batch_size"}}
  #)



