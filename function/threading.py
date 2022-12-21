import threading
import cv2
import numpy as np
import time
from function.processing import *

class MultiThread(threading.Thread):

    def __init__(self, frame, model, bindings, binding_addrs, context, label_name, label_colors, device):
        self._frame = frame
        self._model = model
        self._bindings = bindings
        self._binding_addrs = binding_addrs
        self._context = context
        self._label_names = label_name
        self._label_colors = label_colors
        self._device = device
    
    def run(self):

        # root_box list
        root_box = []

        frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        copy_frame = frame.copy()
        copy_frame, ratio, dwdh = imageProcess(copy_frame, self._device)

        for _ in range(10):
            tmp = torch.randn(1, 3, 640, 640).to(self._device)
            self._binding_addrs['frames'] = int(tmp.data_ptr())
            self._context.execute_v2(list(self._binding_addrs.values()))

        start = time.perf_counter()
        self._binding_addrs['frames'] = int(copy_frame.data_ptr())
        self._context.execute_v2(list(self._binding_addrs.values()))
        end = time.perf_counter()
        print(f"Cost {end - start} s")

        nums = self._bindings['num_dets'].data
        boxes = self._bindings['det_boxes'].data
        scores = self._bindings['det_scores'].data
        classes = self._bindings['det_classes'].data

        boxes = boxes[0, :nums[0][0]]
        scores = scores[0, :nums[0][0]]
        classes = classes[0, :nums[0][0]]

        for box, score, cl in zip(boxes, scores, classes):
            box = postprocess(box, ratio, dwdh).round().int()
            label_name = self._label_names[cl]
            label_color = self._label_colors[cl]

            # found `root` label
            if label_name == "root":
                root_box.append(box)
            label_name += f"{str(round(float(score), 3))}"
            cv2.rectangle(frame, box[:2].tolist(), box[2:].tolist(), label_color, 2)
            cv2.putText(frame, label_name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color, thickness=2)

        return frame, root_box


            





