import torch
import cv2
import numpy as np

def letterbox(frame, new_shape=(640, 640), color=(0, 255, 0), auto=True, scaleup=True, stride=32):
  # Resize and pad image while meeting stride-multiple constrains
  shape = frame.shape[:2]
  if isinstance(new_shape, int):
    new_shape = (new_shape, new_shape)

  # Scale ratio (new / old)
  r = min(new_shape[0] / shape[0], new_shape[1], shape[1])
  if not scaleup:
    r = min(r, 1.0)

  # Compute padding
  new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
  dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

  if auto:
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)

  dw /= 2
  dh /= 2

  if shape[::-1] != new_unpad:
    frame = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
  top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
  left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
  frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
  return frame, r, (dw, dh)

def postprocess(boxes, r, dwdh):
  dwdh = torch.tensor(dwdh*2).to(boxes.device)
  boxes -= dwdh
  boxes /= r
  return boxes

def imageProcess(copy_frame, device):
    copy_frame, ratio, dwdh = letterbox(copy_frame, auto=False)
    copy_frame = copy_frame.transpose((2, 0, 1))
    copy_frame = np.expand_dims(copy_frame, 0)
    copy_frame = np.ascontiguousarray(copy_frame)
    copy_frame = copy_frame.astype(np.float32)

    copy_frame = torch.from_numpy(copy_frame).to(device)
    copy_frame /= 255
    return copy_frame, ratio, dwdh