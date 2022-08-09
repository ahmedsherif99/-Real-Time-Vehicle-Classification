############################################################imports############################################################
import cv2
import numpy as np
import collections
import os

from os.path import isfile, join
##############################################################Constants############################################################.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.2

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)
DARK_BLUE = (255,0,0)

req_index = [2,5,7] #indices of car,truck,bus classes which we will be working with for vehicle classification
det_class = [] #empty list where we will store detected objects

 
##############################image pre-processing#######################################################################################
   
def pre_process(input_image, net):
    
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)
    return outputs

##############################draw labels on bounding boxes#######################################################################################
def draw_label(input_image, label, name, left, top, width, height):
    
    #text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    #dim, baseline = text_size[0], text_size[1]
    
    # Draw bounding rectangle
    cv2.rectangle(input_image, (left, top), ( left + width, top + height),BLUE, THICKNESS)
    
    # Draw classname and confidence score
    if name == 'LMV':
        cv2.putText(input_image, label, (left, top-10), FONT_FACE, FONT_SCALE, RED, 2*THICKNESS)
    elif name == 'HMV':
        cv2.putText(input_image, label, (left, top-10), FONT_FACE, FONT_SCALE, DARK_BLUE, 2*THICKNESS)
 
##############################image post-processing#######################################################################################
  
def post_process(input_image, outputs):
    
    class_ids = []
    confidences = []
    boxes = []
    global det_class
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    

    for output in outputs:
        for row in output:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            
            if class_id in req_index:
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    
                    if (classes_scores[class_id] > SCORE_THRESHOLD):
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        width,height = int(row[2]*image_width) , int(row[3]*image_height)
                        left,top = int((row[0]*image_width)-width/2) , int((row[1]*image_height)-height/2)

                        
                        boxes.append([left, top, width, height])
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        if classes[class_ids[i]] == 'car':
            name = "LMV"
        if classes[class_ids[i]] == 'bus' or classes[class_ids[i]] == 'truck' :
            name = "HMV"
        det_class.append(name)
        
        label = f'{name.upper()} {int(confidences[i]*100)}%'
        draw_label(input_image, label, name, left, top, width, height)
    
    return input_image


##############################Counting vehicles in a frame#############################################

def count_vehicle():

    #calculate the freq of the elements in the detected objects list, 
    #this function returns a dictionary containing the element as key of the dictionary
    #and freq of the element as the value of that particular key.
    freq = collections.Counter(det_class) 
    print(freq)
    # Draw counting texts in the frame
    cv2.putText(img, "LMV:  "+str(freq['LMV']), (20, 40), FONT_FACE, FONT_SCALE, RED, 2*THICKNESS)   
    cv2.putText(img, "HMV:  "+str(freq['HMV']), (20, 60), FONT_FACE, FONT_SCALE, RED, 2*THICKNESS)
    det_class.clear() #clearing the detected objects list to reset the LMV and HMV count

##############################Converting frames to video#############################################
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files.sort(key = lambda x: float(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
########################################--Main function--#############################################

if __name__ == '__main__':

    classesFile = "coco.names"
    classes = None

    with open(classesFile, 'rt') as f:
         classes = f.read().rstrip('\n').split('\n')
        
    ## Model Files
    modelConfiguration = 'yolov3-320.cfg'
    modelWeights= 'yolov3-320.weights'
            
    # configure the network model
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        
    # Configure the network backend
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

##############################For video conversion into sequence of images#############################
   
    #reading the video file
    cap = cv2.VideoCapture('my_vid.mp4')
    success, img = cap.read()
    count = 0
    #looping on all frames of the video and saving each frame after processing
    while success:
        
        detections = pre_process(img, net)
        img = post_process(img.copy(), detections)
        count_vehicle()

        cv2.imwrite("C:/Users/Lenovo/vehicle_classifier/frames/frame%d.jpg" % count, img)     # save frame as JPEG file      
        success,img = cap.read()
        print('Read a new frame: ', success)
        
        count += 1
    

    pathIn= 'C:/Users/Lenovo/vehicle_classifier/frames/'
    pathOut = 'processed_video_yolov3.avi'
    fps = 20.0
    convert_frames_to_video(pathIn, pathOut, fps)
        
   
#############################For static image processing###############################################
   
    #frame = cv2.imread('my_im.jpeg')
    #detections = pre_process(frame, net)
    #img = post_process(frame.copy(), detections)
    #count_vehicle()
    #cv2.imwrite("MY_IM_yolov3.jpg", img)     # save frame as JPEG file  
    #cv2.imshow('Vehicle Classification Output', img)
    #cv2.waitKey(0)
