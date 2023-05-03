import os
import scipy as sp
import scipy.io as sio
from scipy.misc import *
import h5py
from PIL import Image
import numpy as np
import copy




def load_SVHN_train():
    train_data = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])
    
    X_train = []
    for i in range(X.shape[3]):
    
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)


    Y_train = train_data['y']
    
    np.place(Y_train, Y_train == 10, 0)
    return (X_train,Y_train)


def load_SVHN_test():
    test_data = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])
    
    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_data['y']
   
    np.place(Y_test, Y_test == 10, 0)
    
    return (X_test,Y_test)


def sample_info(mat_file, idx):
    sample_info = {}
    file = mat_file
    item = file['digitStruct']['bbox'][idx].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        sample_info = file[item][key]
        values = [file[attr[i].item()][0][0]
                  for i in range(len(sample_info))] if len(sample_info) > 1 else [sample_info[0][0]]
        sample_info[key] = values
    return sample_info



def add_neg_images(img_w_neg,label_w_neg,sample_idx,path):
    path_to_dir, path_to_digit_struct_mat_file = path

    
    path_to_image_file = os.path.join(path_to_dir, f'{sample_idx}.png')
    img = cv2.imread(path_to_image_file)
    if img is None:
        sample_idx = 206
        path_to_image_file = os.path.join(path_to_dir, f'{sample_idx}.png')

    index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1

    
    with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
        sample_info = sample_info(digit_struct_mat_file, index)
        length = len(sample_info['label'])
        sample_left, sample_top, sample_width, sample_height = map(lambda x: [int(i) for i in x],
                                                        [sample_info['left'], sample_info['top'], sample_info['width'], sample_info['height']])
        leftMin, topMin, rightMax, bottomMax = (min(sample_left), min(sample_top),
                                                    max(map(lambda x, y: x + y, sample_left, sample_width)),
                                                    max(map(lambda x, y: x + y, sample_top, sample_height)))

        center_x, center_y, max_side = ((leftMin + rightMax) / 2.0, (topMin + bottomMax) / 2.0, max(rightMax - leftMin, bottomMax - topMin))
        digit_box_left, digit_box_top, digit_box_width, digit_box_height = (center_x - max_side / 2.0, center_y - max_side / 2.0,  max_side, max_side)
    


    # Cropping a random image
    x,y,w,h = digit_box_left, digit_box_top, digit_box_width, digit_box_height
    img_temp = img.copy()

    # selecting an area big enough so it will contain interesting
    #  features and would also fit in the image

    w_c,h_c = 10*w,10*h
    count = 0

    neg_img = img_temp[0:w,0:h]
    found = False
    skip = False
    while not found and count<100:

        count +=1
        x_c= np.random.randint(img_temp.shape[1]-1)
        
        y_c= np.random.randint(img_temp.shape[0]-1)
        
        if count>=100:
            neg_img = img_temp[0:int(y),0:int(x)]
            if neg_img.shape[0] == 0 or  neg_img.shape[1]==0:
                neg_img = img_temp[0:4*w,0:4*h]
                x_c,y_c = 0,0
                w_c,h_c = 2*w,2*h

                if (x <= x_c <= x+w and y<= y_c <=y+h) or (x<= x_c+w_c <=x+w and y<=y_c<=y+h) or \
                    (x<= x_c <=x+w and y<=y_c+h_c<=y+h) or (x<= x_c+w_c <=x+w and y<=y_c+h_c<=y+h)\
                or (x_c <= x <= x_c+w_c and y_c<= y <=y_c+h_c) or (x_c<= x+w <=x_c+w_c and y_c<=y<=y_c+h_c)\
                     or (x_c<= x <=x_c+w_c and y_c<=y+h<=y_c+h_c) or (x_c<= x+w <=x_c+w_c and y_c<=y+h<=y_c+h_c):
                    skip = True
                    found = True
            break

        if (x <= x_c <= x+w and y<= y_c <=y+h) or (x<= x_c+w_c <=x+w and y<=y_c<=y+h) \
            or (x<= x_c <=x+w and y<=y_c+h_c<=y+h) or (x<= x_c+w_c <=x+w and y<=y_c+h_c<=y+h):
            
            continue

        elif (x_c <= x <= x_c+w_c and y_c<= y <=y_c+h_c) or (x_c<= x+w <=x_c+w_c and y_c<=y<=y_c+h_c) \
            or (x_c<= x <=x_c+w_c and y_c<=y+h<=y_c+h_c) or (x_c<= x+w <=x_c+w_c and y_c<=y+h<=y_c+h_c):
            
            continue
        else:
            
            if y_c+h_c > img_temp.shape[0]:
                
                continue
            if x_c+w_c > img_temp.shape[1]:
                continue

            neg_img = img_temp [y_c:y_c+h_c,x_c:x_c+w_c]
            
            if neg_img.shape==(h_c, w_c,3):
                found  = True
                next = False
        
  
    if not skip:
        neg_img = cv2.resize(neg_img, (32,32), interpolation = cv2.INTER_CUBIC)
        neg_img = neg_img[None,...]
        negLabel = np.array([10],dtype=np.uint8)
        img_w_neg = np.vstack((img_w_neg,neg_img))
        label_w_neg = np.vstack((label_w_neg,negLabel))

    return img_w_neg,label_w_neg
        
def add_negative_sampels(digit_images,digit_labels,path_to_dir,\
    path_to_digit_struct_mat_file,data_size):
    with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
        num_samples= len(digit_struct_mat_file['digitStruct']['bbox'])
    negLabel = np.array([10],dtype=np.uint8) 
    img_w_neg = copy.deepcopy(digit_images)
    label_w_neg = copy.deepcopy(digit_labels)
    path = [path_to_dir,path_to_digit_struct_mat_file]
    random_img_idx = np.random.choice(num_samples-1, int(data_size*0.1),replace=False)
    
    for sample_idx in random_img_idx:
        img_w_neg,label_w_neg = add_neg_images(img_w_neg,label_w_neg,sample_idx,path)

    random_gen = np.random.default_rng()
    indices = np.arange(img_w_neg.shape[0])
    random_gen.shuffle(indices)
    img_w_neg_shuffled,label_w_neg_shuffled = img_w_neg[indices],label_w_neg[indices]

    return img_w_neg_shuffled,label_w_neg_shuffled
def save_data(img_w_neg,label_w_neg,dateset_name):
    h5f = h5py.File(f"{dateset_name}", 'w')
    h5f.create_dataset('data', data=img_w_neg)
    h5f.create_dataset('label', data=label_w_neg)
    h5f.close()

if __name__ == '__main__':
    #Dataset location
    train_location = 'data/train_32x32.mat'
    test_location = 'data/test_32x32.mat'

    img_train,label_train = load_SVHN_train()
    img_test,label_test = load_SVHN_test()

    train_size = img_train.shape[0]

    # Creating addtional train data with negative-no digit- examples
    path_to_dir = "data/download/train/"
    path_to_digit_struct_mat_file = os.path.join(path_to_dir, 'digitStruct.mat')

    img_train_w_neg,label_train_w_neg = add_negative_sampels(img_train,label_train,path_to_dir,\
        path_to_digit_struct_mat_file,train_size)

    save_data(img_train_w_neg,label_train_w_neg,dateset_name='train_raw_2.h5')

    # Creating addtional test data with negative-no digit- examples
    path_to_dir = "data/download/test/"
    path_to_digit_struct_mat_file = os.path.join(path_to_dir, 'digitStruct.mat')

    test_size = img_test.shape[0]
    img_test_w_neg,label_test_w_neg = add_negative_sampels(img_test,label_test,path_to_dir,\
        path_to_digit_struct_mat_file,test_size)

    save_data(img_train_w_neg,label_train_w_neg,dateset_name='test_raw_2.h5')

        
