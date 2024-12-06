import sys
sys.path.append("./ByteTrack")
import numpy as np

from yolox.tracker.byte_tracker import BYTETracker # ByteTrack

# from instance import Person, Car, Bike
import torch
from ultralytics import YOLO

###########################################################
#### Person Detector
###########################################################

class YoloByteTrack :
    def __init__(self,cfg) :
        from easydict import EasyDict
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(cfg["model"]).to(self.device)

        self.conf_thresh = cfg["conf_thresh"]
        self.nms_thresh = cfg["nms_thresh"]
        # ByteTracker configs
        '''
        # tracking args
        {
            track_thresh : 0.6 # tracking confidence threshold
            track_buffer : 30, # the frames for keep lost tracks
            match_thresh : 0.9 # matching threshold for tracking
            min-box-area : 100 # filter out tiny boxes
            mot20 : False # test mot20
        }
        '''
        self.tracker_args = EasyDict(cfg["bytetrack"])
        
        self.trackers = [ 
            BYTETracker(self.tracker_args,cfg["bytetrack"]["fps"]),
            BYTETracker(self.tracker_args,cfg["bytetrack"]["fps"]),
            BYTETracker(self.tracker_args,cfg["bytetrack"]["fps"])
        ]

        
    def _init_stats(self):
        pass
    def model_inference(self,input) :
        return self.model(input,conf=self.conf_thresh,verbose=False)
    def __call__(self,frame, frame_wh) :
        '''
        frame : image numpy shape (1,3,h,w)
        '''
        # 0. Primary Detector
        # Input : img_in
        # Return : boxes
        
        predictions = self.model_inference(frame)
        boxes = predictions[0].boxes.xyxy.cpu().numpy()  # Extract bounding box predictions
        confidences = predictions[0].boxes.conf.cpu().numpy()  # Extract confidence scores
        classes = predictions[0].boxes.cls.cpu().numpy()  # Extract class labels
        predictions = np.column_stack((boxes, confidences, classes))
        
        width = frame_wh[0]
        height = frame_wh[1]

        # 1. Track (ByteTrack)
        # Input : boxes
        # Return : boxes with track ID

        dets = [[],[],[]] # person, car, bike
        track_result =  [] # person, car, bike
        
        for i in range(len(predictions)):
            box = predictions[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            cls_conf = box[4]
            cls_id   = int(box[5])
            # if cls_id != 0: # Only for person
            #     continue
            dets[cls_id].append([x1, y1, x2, y2, cls_conf])
        
        for cls_id in [0,1,2] :
            if len(dets[cls_id]) == 0:
                online_targets = self.trackers[cls_id].update(np.empty((0, 5)),(height,width), (height,width))
                
                track_item = []
                for t in online_targets :
                    tlwh = t.tlwh
                    tlbr = t.tlbr
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6

                    if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                        track_item.append([*tlbr,tid,t.score])
                track_result.append(track_item)
                
            else:
                online_targets = self.trackers[cls_id].update(np.array(dets[cls_id]),(height,width), (height,width))
                
                track_item = []
                for t in online_targets :
                    tlwh = t.tlwh # xyxy format
                    tlbr = t.tlbr
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6

                    if tlwh[2] * tlwh[3] > self.tracker_args.min_box_area and not vertical:
                        track_item.append([*tlbr,tid,t.score])
                track_result.append(track_item)
        return track_result
        
