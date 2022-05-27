#!/usr/bin/env python
# coding: utf-8
import numpy as np
import cv2
import glob
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image
import random
import torch
from PIL import Image
from tensorflow.keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
# Load the model
from tensorflow.keras.models import load_model
import argparse
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans

def arg_parse():
    """
    Parse arguements 
    """

    parser = argparse.ArgumentParser(description='player idintification and team recognation ')

    parser.add_argument(
        "--videos", 
        dest='videos', 
        required=True,
        help="video / Directory containing videos to perform detection upon", 
        type=str
    )
    parser.add_argument("--config", default="/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/yolov3.cfg",help="YOLO config path")
    parser.add_argument("--weights", default="/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/yolov3.weights", help="YOLO weights path")
    return parser.parse_args()


# In[7]:


def read_video(video):
    frames_capture=cv2.VideoCapture(video) 
    frames_lst=[]
    while True:
        success,frame = frames_capture.read()
        frames_lst.append(frame)
        if success == False:
            break
    return frames_lst
def select_sample_from_frames(frames, n=100):
    return random.sample(frames, n)




'''
this function is built for k-kmean algorithm

'''
def detect_persons_in_one_frame_for_k_mean(frame):
    if frame is not None:
        height, width, channels =frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
        yolov3.setInput(blob)
        output_layers=yolov3.getUnconnectedOutLayersNames()
        outs = yolov3.forward(output_layers)
        image2=[]
        boxes = []
        confidences = []
        classIDs = []
        crop=[]
    
        for output in outs:
            for detection in output:
                score = detection[5:]
                classID = np.argmax(score)
                confidence = score[classID]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
    
        for i in range (len(boxes)):
        
            if i in idxs :
            
                if classes[classIDs[i]]=='person':
                    x,y,w,h=boxes[i]
                    label= str(classes[classIDs[i]])
                    #cv2.rectangle(frame,(x,y),(x+w,y+h),(220,220,220),1)
                    #cv2.putText(frame,label,(x,y),font,1,(220,220,220),1)
                
                    crop_img = frame[y:y+h, x:x+w]
                    try:
                        img=cv2.resize(crop_img,(100,100))
                        scale=0.50
                        center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
                        width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
                        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
                        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
                        img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]#(80,80,3)
                        img_cropped2=cv2.resize(img_cropped,(64,64))
                   
                    except:
                        continue
                    #arr1 = crop2.reshape((crop2.shape[1]*crop2.shape[0],3))
                    #length, height, depth = crop_img2.shape
                    image2.append(img_cropped2)# reshape it to 2D array to feed it to kmean
                
    
        image2=np.asarray(image2)
        return image2



def yolov5 (img):
    im=array_to_img(img)
    output = model(im)
    results = output.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        con = result['confidence']
        cs = result['class']
        return cs 




def balldetection(frame):
    im=array_to_img(frame)
    output = ballmodel(im)
    results = output.pandas().xyxy[0].to_dict(orient="records")
    for result in results:
        con = result['confidence']
        cs = result['class']
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        font = cv2.FONT_HERSHEY_PLAIN


        label="football"
        cv2.rectangle(frame,(x1, y1), (x2, y2),(160,0,0),2)
        cv2.putText(frame,label,(x1, y1),font,1,(160,0,0),2)
    return frame





'''
this function is built for k-kmean algorithm

'''
def detect_persons_in_one_frame_for_k_mean_pridectwithyolov5(frame):
    FONT = cv2.FONT_HERSHEY_COMPLEX
    # frames from RGB to HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # green mask to select only the field
    mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    
    
    # white + anycolor = anycolor; black + anycolor = black
    frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)
    
    # we use frame_masked to detect player 
    height, width, channels = frame_masked.shape
    blob = cv2.dnn.blobFromImage(frame_masked, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    yolov3.setInput(blob)
    output_layers=yolov3.getUnconnectedOutLayersNames()
    outs = yolov3.forward(output_layers)
    image22=[]
    boxes = []
    confidences = []
    classIDs = []
    crop=[]
    person=[]
    names = []


    for output in outs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIDs.append(class_id)
                names.append(str(classes[class_id]))


                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range (len(boxes)):
        if i in idxs :
            if classes[classIDs[i]]=='person':
                x,y,w,h=boxes[i]
                label= str(classes[classIDs[i]])

                crop_img = frame[y:y+h, x:x+w]
                try:
                    img=cv2.resize(crop_img,(100,100))
                    scale=0.50
                    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
                    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
                    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
                    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
                    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
                    img_cropped2=cv2.resize(img_cropped,(64,64))
                    
                   
                except:
                    continue
                #image2=np.asarray(resiz)
                image22.append(img_cropped2)
                image2=np.asarray(image22)
                person.append([x,y,w,h])
        
        
                representation = intermediate_layer_model.predict(image2)
                representation = representation.reshape(representation.shape[0], -1)
                clusters=kmeans.predict(representation)
                for k in range(len(clusters)):
                    cluster_no=clusters[k]

                    if cluster_no==counter.most_common()[0][0]:
                        x2,y2,w2,h2=person[k]
                        crop_img2 = frame[y2:y2+h2, x2:x2+w2]
                        num=yolov5(crop_img2)
                        if num == None: 
                            label="T2- UN "
                            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (160,0,0), 2)
                            cv2.putText(frame,label,(x2,y2),font,1,(160,0,0),2)
                        else :
                            label="T2- " +str(num)
                            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (160,0,0), 2)
                            cv2.putText(frame,label,(x2,y2),font,1,(160,0,0),2)
                        
                    if cluster_no==counter.most_common()[1][0]:
                        
                        x3,y3,w3,h3=person[k]
                        crop_img2 = frame[y3:y3+h3, x3:x3+w3]

                        num=yolov5(crop_img2)
                        if num ==None :
                            label="T1- UN "
                            cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (220,220,220), 2)
                            cv2.putText(frame,label,(x3,y3),font,1,(220,220,220),2)
                        else :
                            label="T1- "+str(num)
                            cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (220,220,220), 2)
                            cv2.putText(frame,label,(x3,y3),font,1,(220,220,220),2)
    
                    if clusters[k]==counter.most_common()[2][0]:
                        x4,y4,w4,h4=person[k]
                        label="other "
                        cv2.rectangle(frame, (x4, y4), (x4 + w4, y4+ h4), (0,0,0), 2)
                        cv2.putText(frame,label,(x4,y4),font,1,(0,0,0),2) 
                            
                    
    return frame




def main(URL_video):

    capture = cv2.VideoCapture(URL_video)
    print("[PROCESSING] . . . . ")
    crop=[]
    frame_mask=[]
    frames=[]
    image_or=[]
    while capture.isOpened():
        success,frame = capture.read()
        if success == False:
            break
        # frames from RGB to HSV 
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
        # green mask to select only the field
        mask_green = cv2.inRange(hsv, (36, 25, 25), (86, 255, 255))
    
    
        # white + anycolor = anycolor; black + anycolor = black
        frame_masked = cv2.bitwise_and(frame, frame, mask=mask_green)
        frame_mask.append(frame_masked)

        height, width, channels = frame.shape
        arr_kmeans=detect_persons_in_one_frame_for_k_mean_pridectwithyolov5(frame)
        frame=balldetection(frame)

        frames.append(frame)

    capture.release()
    print("[INFO] PROCESSING DONE")
    print("[INFO] saving the result ")
    size = (width,height)

    outt = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    for i in range(len(frames)):
        outt.write((frames[i]))
    outt.release() 
    
args = arg_parse()
print("[INFO] LOAD MODEL ")

# Load Yolo
#Load yolov3.weights and yolov3.cfg (configuration file)
yolov3=cv2.dnn.readNet(args.config, args.weights)

#save COCO object categories in the list 
classes=[]
with open("/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/coco.names","r")as c:
    classes= c.read().splitlines()
layers_names= yolov3.getLayerNames()
colors=np.random.uniform(0,255,size=(len(classes),3))

# load autoencoder model
json_file = open('/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/autoencodermodel_100_epoch_V4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/autoencodermodel_100_epoch_V4.h5")

# load player idintificatio model
model=torch.hub.load('ultralytics/yolov5', 'custom',path='/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/yolov5ID_640_100epoch.pt') 
model.load_state_dict(torch.load('/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/yolov5ID_640_100epoch.pt')['model'].state_dict(),strict=False)

# load foootball detection model
ballmodel=torch.hub.load('ultralytics/yolov5', 'custom',path='/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/balldetection640_100epoch.pt') 
ballmodel.load_state_dict(torch.load('/Users/ragadalhejaily/Documents/finalproject/code/last_project_code/models/balldetection640_100epoch.pt')['model'].state_dict(),strict=False)
print("[INFO] train kmeans for team recognation . ")

capture =read_video(args.videos)
# select_sample_from_frames for train kmeans
sample_of_frames=select_sample_from_frames(capture,n=100)
'''
    this part of code to prepare the frames as training set for k-means
'''
count=0
for frame in sample_of_frames:
    t_x1=detect_persons_in_one_frame_for_k_mean(frame)
    try:
        if t_x1.ndim>1:
            newarr = t_x1 
            
    except:
        continue
    
    if count==0: 
        training_X_for_Kmean=newarr.copy()
    else:
        training_X_for_Kmean=np.concatenate((training_X_for_Kmean, newarr))
    count=count+1
    
'''
 Apply k-means algorithm
'''
intermediate_layer_model=Model(inputs=loaded_model.input,outputs=
                                   loaded_model.get_layer("representation3").output)
representation = intermediate_layer_model.predict(training_X_for_Kmean)
representation = representation.reshape(representation.shape[0], -1)

kmeans = KMeans(n_clusters = 3,random_state=0).fit(representation)
from collections import Counter, defaultdict
counter=Counter(kmeans.labels_)
most_common_element = counter.most_common()

main(args.videos)







