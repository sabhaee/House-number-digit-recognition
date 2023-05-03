# Street View House Number Recognition
 
Project structure is as follows
----------------------------------
- cv_proj.yml # libraries needed to set up enironment
- run.py # code to generate images for grading
- data_prep.py # code to create dataset based on SVHN dataset requires download of .mat files from SVHN website
- classifier.py # contain CNN model and dataloader neede for model inference
- train.py # code for running training and testing
- detect_and_classify.py # main code used in run.py to precess images and detect and classify
- /input_images # image forlder containing images for grading
- /output # directory where image files after detection and annotaion will be saved
- /checkpoints # should contain .pth file for the saved model weights can be downloaded from the provided link ("best_svhn_model_state_MyModel_weights.pth")
- /plots # contains plots from training 
- /data # should contain data for training and testing data can be downloaded from the provided link
-----------------------------------

