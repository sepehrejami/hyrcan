import cv2

def open_default_camera(index=0):
    cap = cv2.VideoCapture(index)
    return cap
