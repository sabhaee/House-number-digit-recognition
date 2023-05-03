import os
import numpy as np
import cv2
import copy
import PIL
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from detect_and_classify import *
# from detect_and_classify import _initialize 
from classifier import SVHN_classifier,SVHN_custom_dataset

IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR="output"
CHKPNT_DIR="checkpoints/"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


# Reference: two functions below for reading and saving videos

def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    retVal = True
    while retVal:
        retVal,frame = video.read()
        yield frame
    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def process_video(ObjDet,video_name,VID_DIR,OUT_DIR):
    
    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    print(image.shape)
   

    hh,ww = image.shape[:-1]
    
    if ww>1000 and hh>1000:
        image = cv2.resize(image, (int(0.25*ww),int(0.25*hh)), interpolation = cv2.INTER_AREA)
    h, w, d = image.shape
 

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    out_path = os.path.join(OUT_DIR, "marked_{}".format(video_name))
    video_out = mp4_video_writer(out_path, (w, h), fps=20)

    frame_num = 1

    while image is not None:

        print("Processing frame {}".format(frame_num))

        result_img = ObjDet.detect_and_classify(image)
        
        video_out.write(result_img)

        image = image_gen.__next__()
        if image is not None:
            hh,ww = image.shape[:-1]
            if ww>1000 and hh>1000:
                image = cv2.resize(image, (int(0.25*ww),int(0.25*hh)), interpolation = cv2.INTER_AREA)

        frame_num += 1

    video_out.release()


def process_image(ObjDet,input_filename,output_filename):

    path_to_image_file = os.path.join(IMG_DIR, input_filename)
    input_image = cv2.imread(path_to_image_file)

    hh,ww = input_image.shape[:-1]
    
    if ww>600 and hh>600:
        input_image = cv2.resize(input_image, (int(0.25*ww),int(0.25*hh)), interpolation = cv2.INTER_AREA)
    # cv2.imshow("input",input_image.astype(np.uint8))
    # cv2.waitKey()
    
    result_image = ObjDet.detect_and_classify(input_image)
    
    cv2.imwrite(os.path.join(OUT_DIR, output_filename), result_image.astype(np.uint8))




if __name__ == '__main__':
    
    checkpoint_name = "best_svhn_model_state_MyModel_weights.pth"

    # model = load_model(model,device, checkpoint_name,CHKPNT_DIR)

    # video_name = "CNN_demo_video.mp4"
    # Please comment out the function below to process video
    # process_video(model,video_name,VID_DIR,OUT_DIR,device)
    
    for i in range(1,6):
        input_filename = f'{i}.png' 
        output_filename = f"output_{i}.png"
        ObjDet = ObjectDetector(checkpoint_name,checkpoint_dir=CHKPNT_DIR)

        process_image(ObjDet,input_filename,output_filename)
    