import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
from torch.utils.data import Dataset
import h5py
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings

from classifier import SVHN_classifier,SVHN_custom_dataset


class ImageClassifier:
    def __init__(self, checkpoint_name,checkpoint_dir="checkpoints/"):
        
        random_seed = 42
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.model = SVHN_classifier()  
        self._load_model()

    def _load_model(self):
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_name)
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.model = self.model.to(self.device)
            self.model.eval()
        except OSError as err:
            print("No checkpoint found - verify checkpoint directory or train model from scratch: {}".format(err))

    def _preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        w,h,c = image.shape
        if w !=32 and h !=32:
            img = cv2.resize(img, (32,32), interpolation = cv2.INTER_CUBIC)

        # Convert the preprocessed image to a tensor
        img_tensor = transform(image)
        img_tensor = img_tensor - torch.mean(img_tensor)
        

        # Move the image tensor to the device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor

    def predict_image(self, image):
        img_tensor = self._preprocess_image(image)
    
        output = self.model(img_tensor)
        sm = torch.nn.Softmax(dim=1)
        prob = sm(output)
        sig = torch.nn.Sigmoid()
        confidence = torch.max(sig(output))
        p, preds = torch.max(prob, dim=1)
        
        return confidence.item(), p.item(), preds.item()
    
class ObjectDetector(ImageClassifier):
    def __init__(self,checkpoint_name,checkpoint_dir="checkpoints/"):
        super().__init__(checkpoint_name,checkpoint_dir)
        

    def _mser_patches(self,regions, region_bbox):
        large_bbox = []
        small_bbox = []
        for i in range(len(regions)):
            x, y, w, h = region_bbox[i]
            
            if w > 32 or h > 32:
                large_bbox.append(region_bbox[i])
            else:
                small_bbox.append(region_bbox[i])

        return large_bbox, small_bbox

    
    def _process_patches(self,image, large_bbox, small_bbox):
        final_windows =[]
        ksize = 48
        # ensuring bboxes have a min dim of ksize and then resize to 32x32
        for i in range(len(large_bbox)):
            x,y,w,h = large_bbox[i]
            
            edge_top = y
            edge_bot = y+h if y+h<=image.shape[0] else image.shape[0]

            height = edge_bot - edge_top 
            if height < ksize:
                
                if edge_bot==image.shape[0]:
                    edge_top -= (ksize-height)
                elif edge_top ==0:                                 
                    edge_bot += (ksize-height)
            height = edge_bot - edge_top 

            edge_left  = x
            edge_right =x+w if x+w<=image.shape[1] else image.shape[1]
            
            width = edge_right - edge_left 
            if width < ksize:
                if width<height and height>=ksize:
                    ksize = h

                if edge_right==image.shape[1]:
                    edge_left -= (ksize-width)
                elif edge_left ==0:
                    edge_left += (ksize-width)
            
            img = image[edge_top : edge_bot , edge_left:edge_right ,:]

            
            final_windows.append([cv2.resize(img, (32,32), interpolation = cv2.INTER_CUBIC),(x,y,w,h)])


        for i in range(len(small_bbox)):
            x,y,w,h = small_bbox[i]
            img = image[y:y+h,x:x+w,:]
            
            left= (ksize - w)//2
            right=(ksize - w)//2 if w%2==0 else (ksize - w)//2 +1
            top = (ksize - h)//2
            bot = (ksize - h)//2 if h%2==0 else (ksize - h)//2 +1
            # npad = ((top,bot),(left,right), (0, 0))
            
            edge_left  = x-left if x-left>=0 else 0
            edge_right =x+w+right if x+w+right<=image.shape[1] else image.shape[1]

            width = edge_right - edge_left 
            if width < ksize:
                if edge_right==image.shape[1]:
                    edge_left -= (ksize-width)
                elif edge_left ==0:
                    edge_left += (ksize-width)

            edge_top = y-top if y-top >=0 else 0
            edge_bot = y+h+bot if y+h+bot<=image.shape[0] else image.shape[0]

            height = edge_bot - edge_top 
            if height < ksize:
                if edge_bot==image.shape[0]:
                    edge_top -= (ksize-height)
                elif edge_top ==0:
                    edge_bot += (ksize-height)

            img = image[edge_top : edge_bot ,\
                edge_left:edge_right ,:]
            
            final_windows.append([cv2.resize(img, (32,32), interpolation = cv2.INTER_CUBIC),(x,y,w,h)])
            
        


        return final_windows

    
    def _patch_evaluation(self,image, patches,level):
        candidates = None
        for patch in patches:
            confidence, prob, _ = self.predict_image(patch[0])
            x, y, w, h = patch[1]
            if confidence>0.8 and prob > 0.9 and float(w)<h and float(h)/w < 2:
                bbox_at_level = np.array([x,y,w,h])*(2**level)

                if candidates is None:
                    candidates = bbox_at_level
                else:
                    candidates = np.vstack((candidates,bbox_at_level))

       
        return candidates
    
    def _find_candidate_ROIs(self,img_temp):
        all_candidates = None
    
        pyramids = [img_temp]
        for i in range(1,5):
            img = pyramids[i-1].copy()
            pyramids.append(cv2.pyrDown(img))

        w,h = img_temp.shape[:-1][::-1]
        
        scaled = [cv2.resize(img_temp, (int(s*w),int(s*h)), interpolation = cv2.INTER_AREA) for s in range(4,1,-2)]
        pyramids= scaled+pyramids


        for level in range(-2, len(pyramids) - 2):
            image_at_level_i = pyramids[level + 2]
            mser = cv2.MSER_create(delta=5, max_variation=0.4, min_diversity=5)

            regions, region_bbox, large_bbox, small_bbox, final_windows, candidates = [], [], [], [], [], []
            image_at_level_i = image_at_level_i.astype(np.uint8)
            gray_img = cv2.cvtColor(image_at_level_i, cv2.COLOR_BGR2GRAY)

            regions, region_bbox = mser.detectRegions(gray_img)

            large_bbox, small_bbox = self._mser_patches(regions, region_bbox)
            final_windows = self._process_patches(image_at_level_i, large_bbox, small_bbox)

            candidates = self._patch_evaluation(image_at_level_i, final_windows,level)
            if candidates is not None:
                if all_candidates is None:
                    all_candidates = candidates
                else:
                    all_candidates = np.vstack((all_candidates, candidates))

        return all_candidates
    
    @staticmethod
    def _dist(p1,p2):
        x_1,y_1 = p1
        x_2,y_2 = p2 
        
        return np.sqrt((x_1-x_2)**2 + (y_1-y_2)**2)
    
    def _bbox_clustering(self, box_vector):
        kmeans_diff_ks = []

        if box_vector.shape[0] > 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')

                kmeans_diff_ks = [KMeans(n_clusters=k, random_state=42).fit(box_vector)
                                  for k in range(2, min(6, box_vector.shape[0]))]

            k_values = np.arange(2, box_vector.shape[0])
            silhouette_scores = []
            bbox_final = []
            k = 0

            for h in kmeans_diff_ks:
                if len(np.unique(h.labels_)) > 1:
                    silhouette_scores.append(silhouette_score(box_vector, h.labels_))
                else:
                    k = 1
                    break

            if k != 1:
                if len(silhouette_scores) == 1:
                    bbox_final = kmeans_diff_ks[0].cluster_centers_
                else:
                    idx_best_2_ks = np.argpartition(silhouette_scores, -2)[-2:]
                    k = max(k_values[idx_best_2_ks])
                    bbox_final = kmeans_diff_ks[np.where(k_values == k)[0][0]].cluster_centers_
            else:
                bbox_final = kmeans_diff_ks[0].cluster_centers_
        else:
            bbox_final = box_vector

        return bbox_final
    def _bbox_non_max_suppression(self, input_image, final_groups):
        boxes = []
        score = []
        confidence = []
        idxs = []
        temp_img = input_image.copy()
        for i in range(final_groups.shape[0]):
            for j in range(final_groups[i].shape[0]):
                x,y,w,h= final_groups[i][j]
                x1,y1,x2,y2 = x,y,x+w,y+h
                
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                left = int(0.3*w)
                right= int(0.3*w)
                top  = int(0.3*h)
                bot  = int(0.3*h)

                edge_left  = x-left if x-left>=0 else 0
                edge_right =x+w+right if x+w+right<=temp_img.shape[1] else temp_img.shape[1]

                
                edge_top = y-top if y-top >=0 else 0
                edge_bot = y+h+bot if y+h+bot<=temp_img.shape[0] else temp_img.shape[0]


                img = temp_img[edge_top : edge_bot ,edge_left:edge_right ,:]
                patch = cv2.resize(img, (32,32), interpolation = cv2.INTER_CUBIC)
                
                
                cert,prob, prediction = self.predict_image(patch)
                if cert>0.5 and prob>0.5:
                    boxes.append([x1,y1,x2,y2])
                    score.append(prob)
                    idxs.append(prediction)
                    confidence.append(cert)


        boxes = torch.Tensor(boxes)
        score = torch.Tensor(score)
        idxs = torch.Tensor(idxs)
        confidence = torch.Tensor(confidence)
        # NMS
        bbox_final_idx = torchvision.ops.batched_nms(boxes,score,idxs,iou_threshold=0.5)

        nms_bbox =torch.index_select(boxes, 0, bbox_final_idx)
        results_nms = torch.index_select(idxs, 0, bbox_final_idx)
        score_nms = torch.index_select(score, 0, bbox_final_idx)
        confidence_nms = torch.index_select(confidence, 0, bbox_final_idx)

        return nms_bbox,results_nms,score_nms,confidence_nms 



    def _evaluate_candidate_bboxes_nms(self,input_image,bbox_final,score_nms,results_nms):
        img_temp = input_image.copy()
        not_bbox_idx = []
        bbox_idx = []
        if bbox_final.ndim>1:
            area = (bbox_final[:,2]-bbox_final[:,0])*(bbox_final[:,3]-bbox_final[:,1])
            avg_area = torch.mean(area).item()

            for i in range(bbox_final.shape[0]):
                area_i = area[i].item()
                if area_i < 0.1 * avg_area:
                    if i not in not_bbox_idx:
                        not_bbox_idx.append(i)
                for j in range(bbox_final.shape[0]):
                
                    if i !=j:
                        if results_nms[i] == 10:
                                not_bbox_idx.append(i)
                        if results_nms[j] == 10:
                                    not_bbox_idx.append(j)

                        x1_i,y1_i,x2_i,y2_i = bbox_final[i]

                        x1_j,y1_j,x2_j,y2_j = bbox_final[j]
                        
                        
                        area_j = area[j].item()
                        # if i is inside j
                        if x1_i>=x1_j and y1_i>=y1_j \
                            and x2_i<=x2_j and y1_i<=y2_j:
                            if area_j >1.5*avg_area:
                                if j not in not_bbox_idx:
                                        not_bbox_idx.append(j)
                            else:
                                if i not in not_bbox_idx:
                                    not_bbox_idx.append(i)

                        elif abs(x1_i - x1_j) <= 10 or abs(x2_i - x2_j) <= 10:
                            if i not in not_bbox_idx and  j not in not_bbox_idx:
                                if score_nms[i] < score_nms[j]:
                                    if i not in not_bbox_idx:
                                        not_bbox_idx.append(i)
                                elif score_nms[i] > score_nms[j]:
                                    if j not in not_bbox_idx:
                                        not_bbox_idx.append(j)

                
        indxs = np.arange(bbox_final.shape[0])

        for idx in indxs:
            if idx not in not_bbox_idx:
                bbox_idx.append(idx)
        return bbox_final[bbox_idx],score_nms[bbox_idx],results_nms[bbox_idx]
        
    
    def _detect_and_anotate_nms (self,input_image,nms_bbox,results_nms,score_nms,condifence_nms):
    
        display_img = input_image.copy()
    
        for i in range(nms_bbox.shape[0]):
            x1,y1,x2,y2 = nms_bbox[i]
            prediction = int(results_nms[i].item())

            
            prob = score_nms[i].item()
            cert = condifence_nms[i].item()
            
            if prediction != 10:
                cv2.rectangle(display_img, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 1)
                cv2.putText(display_img, "{}".format(prediction), (int(x1), int(y1)+3), cv2.FONT_HERSHEY_PLAIN, 2, (0, 50, 255), 2)
            
        
        # cv2.imshow("final",display_img)
        # cv2.waitKey()

        return display_img

    def detect_and_classify(self,input_image):
        img_temp = np.copy(input_image)

        # Find candidate regions of interest
        candidates = self._find_candidate_ROIs(img_temp)

        # Perform final detection and classification on candidate regions
        if len(candidates)>0:
            grouped_bbox = self._bbox_clustering(candidates)

        buckets = [[] for i in range(len(grouped_bbox))]
        cluster_centers = grouped_bbox[:,0:2]
        for bbox in candidates:
            dist_to_cluster = np.array([self._dist(bbox[:2],grouped_bbox[i,:2]) for i in range(len(grouped_bbox))])
            

            group_id = np.argmin(dist_to_cluster)
            buckets[group_id].append(bbox)

        new_groups = [self._bbox_clustering(np.array(bbox_groups)) for bbox_groups in buckets if len(bbox_groups)!=0]
        new_groups = [g for g in new_groups if len(g)>0]


        in_group_max_dist =  []
        for g in new_groups:
            dist_temp = []
            g = np.array(g)
            for i in range(len(g)):
                for j in range(len(g)):
                    dist_temp.append(self._dist(g[i,0:2],g[j,0:2]))
                    

            in_group_max_dist.append(max(dist_temp))

        in_group_max_dist=np.array(in_group_max_dist)

        new_groups = np.array(new_groups,dtype=object )

        # removing the clusters with wide distribution 
        final_groups =new_groups[in_group_max_dist<70]


        nms_bbox,results_nms,score_nms,condifence_nms  = self._bbox_non_max_suppression(img_temp,final_groups)
        
        nms_bbox,score_nms,results_nms = self._evaluate_candidate_bboxes_nms(img_temp,nms_bbox,score_nms,results_nms)
    
        
        final_image = self._detect_and_anotate_nms (img_temp,nms_bbox,results_nms,score_nms,condifence_nms)

        return final_image


            