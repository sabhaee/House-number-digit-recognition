import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from classifier import SVHN_classifier, SVHN_custom_dataset, DeviceDataLoader
from torchvision import models
import h5py
import matplotlib.pyplot as plt

class SVHNTrainer:
    def __init__(self,device,data_dir="data/",check_dir = "checkpoints/" ):
        self.DATA_DIR = data_dir
        self.CHKPNT_DIR = check_dir
        self.device = device
        self.model = None
        self.optimizer = None

    def to_device(self,data):
        if isinstance(data, (list,tuple)):
            return [self.to_device(x, self.device) for x in data]
        return data.to(self.device, non_blocking=True)
    
    def intialize_model(self,model_architecture,chkpnt_name,lr=1e-4,load_checkpoint = True):
        # Model architecture:{'Mymodel','VGG16_pretrained','VGG16'}
        if model_architecture=='Mymodel':
            model = SVHN_classifier()
        else:
            model = models.vgg16(pretrained=True)
            

            if model_architecture =='VGG16_pretrained':
                for param in model.features.parameters():
                    param.requires_grad = False
                    num_ftrs = model.classifier[6].in_features
                    model.classifier[6] = nn.Linear(num_ftrs,11)
                    params_to_update = model.parameters()
                    params_to_update = []
                for name,param in model.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
            
            else:
                for param in model.features.parameters():
                    param.requires_grad = True
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs,11)
                params_to_update = model.parameters()


        learning_rate = lr#1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model = self.to_device(model,self.device)
        
        history = []
        last_epoch = 0
        # Set parameter below to True to load the checkpoint. Please make sure checkpoint is downloaded and saved in the
        # cehckpoint directory
        load_from_checkpoint = load_checkpoint# True
        if load_from_checkpoint:
            try:
                checkpoint = torch.load(os.path.join(self.CHKPNT_DIR, f"svhn_{chkpnt_name}_checkpoint.pth"))
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                last_epoch = checkpoint['epoch']
                history += checkpoint['history']
            except OSError as err:
        
                print("No checkpoint found - verify CHKPNT_DIR else train model from scrath: {}".format(err))
        else:
            last_epoch = 0



        self.to_device(model, self.device)
        
        self.model = model
        self.optimizer = optimizer

        return history,last_epoch
    
    def evaluate_model(self,val_dataLoader,dataset_size,phase="validation"):
        model = self.model
        optimizer = self.optimizer
        if phase == 'test':
            optimizer = None
        
        model.train(False)
        model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        val_dataset_size = dataset_size[phase]
        
        for batch in val_dataLoader: 
            images, targets = batch  
            # One hot encoding for losses other than CrossEntropy
            targets = F.one_hot(targets,11).float()

            # subtracting the mean from each image
            images=images-torch.mean(images)

            if optimizer is not None:
                optimizer.zero_grad()

            with torch.no_grad():
                model_outputs = model(images)
                batch_loss = nn.BCEWithLogitsLoss()(model_outputs , targets)
            
                _, predictions = torch.max(model_outputs, dim=1)

            
            running_loss += batch_loss.item() * images.size(0)
            # for one hot encoded case
            _, targets = torch.max(targets, dim=1)
            running_corrects += torch.sum(predictions == targets.data)

        epoch_loss = running_loss / val_dataset_size
        epoch_acc = running_corrects.double() / val_dataset_size
    
    
        return {'val_loss': epoch_loss, 'val_accuracy': epoch_acc.item()}

    def _create_dataloader(self,data_path,transform,batch_size,phase = 'train'):
        dataloader = []
        if phase =='train':
            dataset = SVHN_custom_dataset(data_path,transform)
            
            # Hyperparameter: spliting training data into 80/20 train, validation set
            validation_size = 17000
            train_size = len(dataset) - validation_size

            samples = random_split(dataset,[train_size,validation_size])

            train_samples = samples[0]
            validation_samples = samples[1]

            dataset_size ={"train": train_size,"validation":validation_size,"test":None}

            # Hyperparameter: Batch size
            batch_size = batch_size 

            # TODO: num worker need to be change for cpu
            train_dataLoader = DataLoader(train_samples, batch_size, shuffle=True, num_workers=2)
            val_dataLoader = DataLoader(validation_samples, batch_size*2, num_workers=2)

            train_dataLoader = DeviceDataLoader(train_dataLoader, device)
            val_dataLoader = DeviceDataLoader(val_dataLoader, device)
            
            dataloader = [train_dataLoader,train_dataLoader]

        elif phase == 'test':
            # batch_size = batch_size# 128 
            dataset_test = SVHN_custom_dataset(data_path,transform)
            dataset_size ={"train": None,"validation":None,"test":len(dataset_test)}
            test_loader = DataLoader(dataset_test, batch_size)#, num_workers=2)#, pin_memory=True)
            test_loader = DeviceDataLoader(test_loader, self.device)
            dataloader = [test_loader]
        else:
            raise ValueError("INCORECT PHASE SELECTION: please select from the following list: \n {'train','test'}")


        return dataloader , dataset_size

            
    
    def train(self,history,last_epoch,data_path,model_name,transform,epochs,batch_size = 64): 

        dataloader,dataset_size = self._create_dataloader(self,data_path,transform,batch_size,phase = 'test')
        train_dataLoader,val_dataLoader = dataloader
        # Hyperparameter: Numebr of epochs and learning rate
        num_epochs = epochs#50
        num_epochs += last_epoch
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

        

        model = self.model
        optimizer = self.optimizer

        train_history = history
        best_accuracy = 0

        for epoch in range(last_epoch,epochs):
            # Trianining
            model.train(True) # Setting model to training mode
            

            running_loss = 0.0
            running_corrects = 0
            train_dataset_size = dataset_size["train"] #63257

            for batch in train_dataLoader:
                images, targets = batch 

                # One hot encoding for losses other than CrossEntropy
                targets = F.one_hot(targets,11).float()

                # subtracting the mean from each image
                images=images-torch.mean(images)
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    out = model(images)
                    loss = nn.BCEWithLogitsLoss()(out , targets)
                    
                    _, predictions = torch.max(out, dim=1)

                    loss.backward()
                    optimizer.step()
            
                
                running_loss += loss.item() * images.size(0)
                # for one hot encoded case
                _, targets = torch.max(targets, dim=1)
                running_corrects += torch.sum(predictions == targets.data)
                
            epoch_loss = running_loss / train_dataset_size
            epoch_acc = running_corrects.double() / train_dataset_size

            # Validation phase
            result = self.evaluate_model(model,optimizer, val_dataLoader,dataset_size,phase="validation")
            result['train_acc'] = epoch_acc.item() 
            result['train_loss'] = epoch_loss 

            #############################################
            # printing training history
          
            train_history.append(result)
            
            checkpoint = {
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': train_history
            }
            
            os.makedirs(self.CHKPNT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(
                self.CHKPNT_DIR, f"svhn_{model_name}_checkpoint.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            
            if result['val_accuracy'] > best_accuracy:
                checkpoint_path = os.path.join(
                self.CHKPNT_DIR, 'best_svhn_model_state.pth')
                torch.save(checkpoint, checkpoint_path)
            
                best_accuracy = result['val_accuracy']

            # printing training history
            print(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss: {epoch_loss:.4f}, "
                f"Train Acc: {epoch_acc:.4f}, "
                f"Validation Loss: {result['val_loss']:.4f}, "
                f"Validation Acc: {result['val_accuracy']:.4f}"
            )    
            # scheduler.step(result["val_loss"])
        print("Training completed!")
        
        
        self.plot_accuracy_curve(history)
        self.plot_loss_curve(history)

        return history 

    def test(self,data_path,transform,batch_size = 128):
        
        dataloader,dataset_size = self._create_dataloader(self,data_path,transform,batch_size,phase = 'test')
        test_loader = dataloader[0]
        # result = evaluate_model(model, test_loader)
        
        result = self.evaluate_model(test_loader,dataset_size,phase="test")

        return result

    def plot_accuracy_curve(history):
        valid_accuracy = []
        train_accuracy=[]
        for epoch in history:
            valid_accuracy.append(epoch['validation_accuracy'])
            train_accuracy.append(epoch['train_acc'])
        plt.plot(train_accuracy)
        plt.plot(valid_accuracy)
        plt.legend(['Training', 'Validation'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

    def plot_loss_curve(history):
        trainig_loss = []
        valid_loss = []
        for epoch in history:
            trainig_loss.append(epoch['train_loss'])
            valid_loss.append(epoch['val_loss'])
        plt.plot(trainig_loss)
        plt.plot(valid_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Training', 'Validation'])


if __name__ == "__main__":
    DATA_DIR = "data/"
    CHKPNT_DIR = "checkpoints/"
    
    # Model architecture can be selected from between the following options
    # Model architecture: {'Mymodel', 'VGG16_pretrained', 'VGG16'}
    model_architecture = "Mymodel"

    if model_architecture == "Mymodel":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        checkpoint_name = "Mymodel_train_lr1e-4_BCE"
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(244),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if model_architecture == "VGG16_pretrained":
            checkpoint_name = "VGG16_pretrain_train_lr1e-6_BCE"
        elif model_architecture == "VGG16":
            checkpoint_name = "VGG16_train_lr1e-6_BCE"
        else:
            raise ValueError(
                "INCORRECT MODEL ARCHITECTURE SELECTION: Please select from the following list: \n {'Mymodel', 'VGG16_pretrained', 'VGG16'}"
            )
    
    random_seed = 9
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    SVHNModel = SVHNTrainer(DATA_DIR, CHKPNT_DIR)

    model,history,last_epoch = SVHNModel.initialize_model(
        model_architecture, 
        checkpoint_name, 
        lr=1e-4, 
        load_checkpoint=True
    )

    print(len(history))
    print(last_epoch)

    """
    Train
    """
    # path_train = os.path.join(DATA_DIR,"train_32x32_w_neg_ex.h5")
    # train_history,last_epoch = SVHNModel.train(
    #     history, 
    #     last_epoch, 
    #     path_train, 
    #     model_architecture,
    #     transform,
    #     epochs=50,
    #     batch_size = 64
    # )

    """
    Test
    """
    path_test = os.path.join(DATA_DIR, "test_32x32_w_neg_ex.h5")
    test_results = SVHNModel.test(
        path_test,
        transform,
        batch_size = 128)
    

    print(test_results) 