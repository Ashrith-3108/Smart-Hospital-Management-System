import cv2
import numpy as np
import pytesseract
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

labels = ['Aadhar_no', 'DOB', 'Gender', 'Name']
#yolo confidence threshold to detect hand signs
CONFIDENCE_THRESHOLD = 0.50
GREEN = (0, 255, 0)
yolo_model = YOLO("model/best.pt")
print("Yolo11 Model Loaded")

#function to detect aadhar card
def getAadharNo(frame):
    global yolo_model
    detections = yolo_model(frame)[0]
    label = None
    aadhar_no = ""
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = labels[int(cls_id)]
            if label == 'Aadhar_no':
                region = frame[ymin:ymax, xmin:xmax]
                text = pytesseract.image_to_string(region)
                if len(text.strip()) > 0:
                    aadhar_no = text.strip()
                    break                       
    return aadhar_no

def maskAadhar(img):
    global yolo_model
    detections = yolo_model(img)[0]
    label = None
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        cls_id = data[5]
        if float(confidence) >= CONFIDENCE_THRESHOLD:
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])            
            label = labels[int(cls_id)]
            if label == 'Aadhar_no':
                max_mask = xmin + int(xmax / 3)
                for y in range(ymin, ymax):
                    for x in range(xmin, max_mask):
                        img[y, x] = [0, 0, 0]    
            cv2.rectangle(img, (xmin, ymin) , (xmax, ymax), GREEN, 2)
            cv2.putText(img, label, (xmin, ymin-10),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)        
    return img


img = cv2.imread("samples/IMG20211026130851.jpg")
img = cv2.resize(img, (500, 500))
aadhar_no = getAadharNo(img)
print(aadhar_no)
img = maskAadhar(img)
cv2.imshow("aa", img)
cv2.waitKey(0)

'''
img = cv2.imread("samples/download1.jpg")
img = cv2.resize(img, (500, 500))

boxes = []
#boxes.append([134, 397, 225, 424]) amee
#boxes.append([94, 255, 260, 290])

#boxes.append([159,301,356,374])download

#download1
boxes.append([174, 414, 272, 426])
boxes.append([162, 268, 340, 291])

for i in range(len(boxes)):
    box = boxes[i]
    region = img[box[1]:box[3], box[0]:box[2]]
    text = pytesseract.image_to_string(region)
    print(text)
    cv2.imshow("region", region)
    cv2.waitKey(0)
    max_mask = box[0] + int(box[2] / 3)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    for y in range(box[1], box[3]):
        for x in range(box[0], max_mask):
            img[y, x] = [0, 0, 0]    

cv2.imshow("aa", img)
cv2.waitKey(0)
'''
