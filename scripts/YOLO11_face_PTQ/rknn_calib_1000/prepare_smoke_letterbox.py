import cv2
im = cv2.imread('/mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/rknn_calib_1000/wider_val_0000.jpg')
print('image shape:', im.shape)
h, w = im.shape[:2]
scale = min(640 / w, 640 / h)
nw, nh = round(w * scale), round(h * scale)
resized = cv2.resize(im, (nw, nh))
canvas = cv2.copyMakeBorder(resized, (640 - nh) // 2, 640 - nh - (640 - nh) // 2, (640 - nw) // 2, 640 - nw - (640 - nw) // 2, cv2.BORDER_CONSTANT, value=(114, 114, 114))
cv2.imwrite('/mnt/d/yolo_rk3576_deploy/scripts/YOLO11_face_PTQ/rknn_calib_1000/smoke_letterbox.jpg', canvas)
