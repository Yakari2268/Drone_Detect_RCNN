import torch
import sys
import cv2
import numpy as np
sys.path.insert(0, './yolov5')  # Adjust path to YOLOv5 repo if needed

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

def letterbox(im, new_shape=640, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)


# Load model
device = select_device('cuda:0')  # change to 'cuda:0' for GPU
model = DetectMultiBackend('yolov5s.pt', device=device)
stride = model.stride
names = model.names
img_size = 640

def detect_camera():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        img = letterbox(frame, img_size, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
    
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0  # Normalize 0 - 1
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
    
        # Inference
        pred = model(img_tensor)
        pred = non_max_suppression(pred, 0.25, 0.45)
    
        # Process detections
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
    
        # Display
        cv2.imshow('YOLOv5 Camera', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()

