import cv2
import threading
import torch
import numpy as np
from collections import OrderedDict, namedtuple
from function.threading import MultiThread

# check cuda
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    print("cuda is not defined")

# TensorRT path
MODEL_PATH = "TensorRT model path"

# TensorRT settings
binding = namedtuple("binding", ("name", "dtype", "shape", "data", "ptr"))
logger = trt.Logger(trt.Logger.INFO)
trt.init_livnvinfer_plugins(logger, namespace="")

# Reading TensorRT file
with open(MODEL_PATH, "rb") as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())

# Binding settings
bindings = OrderedDict()
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
    bindings[name] = binding(name, dtype, shape, data, int(data.data_ptr()))
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
context = model.create_execution_context()

# Setting labelname and label colors
label_name = ["root", "sugarcane", "weed"]
label_colors = [[255, 0, 255], [0, 255, 0], [255, 0, 0]]


cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != 27:
    frame = cap.read()

    if frame is None:
        print("frame is not difined")
        break
    
    print(threading.activeCount())
    if(threading.activeCount() == 1):
        th = MultiThread(frame, model, binding_addrs, context, label_name, label_colors, device)
        frame, root_box = th.run()

        print(root_box)

    cv2.imshow("inferencing", frame)

cap.release()
cv2.destroyAllWindows()