import cv2
import time
import numpy as np
import sys
sys.path.append("../")
from train_models.MTCNN_config import config
from nms import py_nms

class MtcnnDetector(object):
    def __init__(self,
                 detectors,
                 min_face_size=24,
                 stride=2,
                 threshold=[0.6, 0.6, 0.9],
                 scale_factor=0.79
                 ):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.train_face = config.train_face
        self.train_mode = config.train_mode
        if self.train_face:
            if self.train_mode ==1:
                self.nms_thresh = [0.5,0.6,0.6]
            else:
                self.nms_thresh = [0.3,0.3,0.3]
        else:
            self.nms_thresh = [0.5,0.6,0.6]
        self.scale_factor = scale_factor
        self.r_out = config.r_out        

    def convert_to_square(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def convert_to_rect(self, bbox):
        """
            convert bbox to square
        Parameters:
        ----------
            bbox: numpy array , shape n x 5
                input bbox
        Returns:
        -------
            square bbox
        """
        rect_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        h_n = np.maximum(np.maximum(h, w),2)
        w_n = h_n/2
        rect_bbox[:, 0] = bbox[:, 0] + w * 0.5 - w_n * 0.5
        rect_bbox[:, 1] = bbox[:, 1] + h * 0.5 - h_n * 0.5
        rect_bbox[:, 2] = rect_bbox[:, 0] + w_n - 1
        rect_bbox[:, 3] = rect_bbox[:, 1] + h_n - 1
        return rect_bbox

    def calibrate_box(self, bbox, reg,height,width):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """
        if config.Debug:
            print("shape ",height,width)
        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        if config.Debug:
            print("x1 ",bbox_c[:,0])
            print("y1 ",bbox_c[:,1])
            print("x2 ",bbox_c[:,2])
            print("y2 ",bbox_c[:,3])
        keep = np.where(bbox_c[:,0] >0)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,1] >0)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,2] <width)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,3] <height)
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,2] > bbox_c[:,0])
        bbox_c = bbox_c[keep]
        keep = np.where(bbox_c[:,3] > bbox_c[:,1])
        bbox_c = bbox_c[keep]
        return bbox_c

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map
        Parameters:
        ----------
            cls_map: numpy array , n x m 
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        #stride = 4
        cellsize = 12
        #cellsize = 25

        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T
    #pre-process images
    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        #img_resized = img_resized / 127.5 -1.0
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        keep = np.where(bboxes[:,0]< w)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,1]< h)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,2] >0)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,3] >0)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,2] > bboxes[:,0])
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,3] > bboxes[:,1])
        bboxes = bboxes[keep]
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
    
    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        net_size = 12
        
        current_scale = float(net_size) / self.min_face_size  # find initial scale
        # print("current_scale", net_size, self.min_face_size, current_scale)
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            #return the result predicted by pnet
            #cls_cls_map : H*w*2
            #reg: H*w*4
            cls_cls_map, reg = self.pnet_detector.predict(im_resized)
            #boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
            #print("in MtCnnDetector pnet out shape ",cls_cls_map.shape, reg.shape)
            #cls_map = cls_cls_map[:,:,1]
            #print("scale, threshold ",current_scale,self.thresh[0])
            #boxes = self.generate_bbox(cls_map,reg,current_scale,self.thresh[0])
            boxes = self.generate_bbox(cls_cls_map[:,:,1], reg, current_scale, self.thresh[0])

            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape
            if boxes.size == 0:
                continue
            keep = py_nms(boxes[:, :5], self.nms_thresh[0])
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return  None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], self.nms_thresh[0])
        all_boxes = all_boxes[keep]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        #print('pnet box ',np.shape(all_boxes))
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return  boxes_c,all_boxes
    def detect_rnet(self, im, dets):
        """Get face candidates using rnet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of pnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        height,width = h,w
        if self.train_face:
            dets = self.convert_to_square(dets)
        else:
            #dets = self.convert_to_rect(dets)
            dets = dets
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        #num_boxes = dets.shape[0]
        num_boxes = tmpw.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        if num_boxes <= 0:
            return None,None
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24))-127.5) / 128
        #cls_scores : num_data*2
        #reg: num_data*4
        #landmark: num_data*10
        if self.train_face:
            #cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
            cls_scores, reg, landmark = self.rnet_detector.predict(cropped_ims)
        else:
            cls_scores, reg = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            if self.train_face:
                landmark = landmark[keep_inds]
        else:
            return  None, None        
        if self.train_face:
            #width
            w = boxes[:,2] - boxes[:,0] + 1
            #height
            h = boxes[:,3] - boxes[:,1] + 1
            landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
            landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T 
        #"Minimum"
        if self.r_out:
            keep = py_nms(boxes, self.nms_thresh[1],"Minimum")
        else:
            keep = py_nms(boxes, self.nms_thresh[1],"Union")
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep],height,width)
        if self.train_face:
            landmark = landmark[keep]
            return boxes_c,landmark
        else:
            return  boxes_c,None
    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        height,width = h, w
        if self.train_face:
            dets = self.convert_to_square(dets)
        else:
            #dets = self.convert_to_rect(dets)
            dets = dets
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        real_box_num = 0
        for i in range(num_boxes):
            if tmph[i]<=1 or  tmpw[i]<=1:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48))-127.5) / 128
            real_box_num+=1
        if real_box_num <=0:
            return  None, None
        if self.train_face:
            cls_scores, reg,landmark = self.onet_detector.predict(cropped_ims)
        else:  
            cls_scores, reg = self.onet_detector.predict(cropped_ims)
        #prob belongs to face
        cls_scores = cls_scores[:,1]        
        keep_inds = np.where(cls_scores > self.thresh[2])[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            if self.train_face:
                landmark = landmark[keep_inds]
        else:
            return  None, None
        
        #width
        w = boxes[:,2] - boxes[:,0] + 1
        #height
        h = boxes[:,3] - boxes[:,1] + 1
        if self.train_face:
            landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes[:,0],(5,1)) - 1).T
            landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes[:,1],(5,1)) - 1).T        
        boxes_c = self.calibrate_box(boxes, reg,height,width)         
        keep = py_nms(boxes_c,self.nms_thresh[2], "Minimum")
        boxes_c = boxes_c[keep]
        if self.train_face:
            landmark = landmark[keep]
            return boxes_c,landmark
        else:
            return boxes_c,None
    #use for video
    def detect(self, img):
        """Detect face over image
        """
        boxes = None
        t = time.time()
    
        # pnet
        t1 = 0
        if self.pnet_detector:
            boxes_c,all_boxes = self.detect_pnet(img)
            #boxes_c = all_boxes
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t1 = time.time() - t
            t = time.time()
            print("Pnet out ",boxes_c.shape)
            order_idx = np.argsort(boxes_c[:,4])[:-1]
            sel_num = config.P_Num if len(boxes_c) > config.P_Num else len(boxes_c)
            boxes_c = boxes_c[order_idx[:sel_num]]
            boxes_p = boxes_c
    
        # rnet
        '''
        for i in range(10):
            print("box_c ",map(int,boxes_c[i]))
            print("box",map(int,all_box[i]))
        '''
        t2 = 0
        if self.rnet_detector:
            boxes_c,landmark_r = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
            t2 = time.time() - t
            t = time.time()
            bbox_r = boxes_c
        if self.r_out:
            print("time cost " + '{:.3f}'.format(t1 + t2) + '  pnet {:.3f}  rnet {:.3f} '.format(t1, t2)) 
            return bbox_r,landmark_r
        
        # onet 
        t3 = 0
        if self.onet_detector:
            #boxes, boxes_c,landmark = self.detect_onet(img, boxes_c)
            if config.o_out:
                boxes_c,landmark = self.detect_onet(img, boxes_p)
            else:
                boxes_c,landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
    
            t3 = time.time() - t
            t = time.time()
            print(
                "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))                                                                                
        return boxes_c,landmark

    def detect_face(self, test_data):
        all_boxes = []#save each image's bboxes
        landmarks = []
        batch_idx = 0
        sum_time = 0
        #test_data is iter_
        data_num = test_data.size
        print("MtcnnDetect image num ",data_num)
        #for databatch in test_data:
        for i in range(data_num):
            databatch = test_data.next()
            #databatch(image returned)
            if batch_idx % 100 == 0:
                print("%d images done" % batch_idx)
            im = databatch
            # pnet
            t1 = 0
            if self.pnet_detector:
                t = time.time()
                #ignore landmark 
                boxes_c, landmark = self.detect_pnet(im)
                t1 = time.time() - t
                sum_time += t1
                if boxes_c is None:
                    print("img path: ",test_data.img_path)
                    print("boxes_c is None...")
                    all_boxes.append(np.array([]))
                    #pay attention
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue
                order_idx = np.argsort(boxes_c[:,4])[:-1]
                sel_num = config.P_Num if len(boxes_c) > config.P_Num else len(boxes_c)
                boxes_c = boxes_c[order_idx[:sel_num]]
            # rnet
            t2 = 0
            if self.rnet_detector:
                t = time.time()
                #ignore landmark                 
                boxes_c, landmark = self.detect_rnet(im, boxes_c)
                t2 = time.time() - t
                sum_time += t2
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))
                    batch_idx += 1
                    continue
            # onet
            t3 = 0
            if self.onet_detector:
                t = time.time()
                boxes_c, landmark = self.detect_onet(im, boxes_c)
                t3 = time.time() - t
                sum_time += t3
                if boxes_c is None:
                    all_boxes.append(np.array([]))
                    landmarks.append(np.array([]))                    
                    batch_idx += 1
                    continue
                print(
                    "time cost " + '{:.3f}'.format(sum_time) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))
                                                                                                                    
                                                                                                                   
            all_boxes.append(boxes_c)
            landmarks.append(landmark)
            batch_idx += 1
        #num_of_data*9,num_of_data*10
        return all_boxes,landmarks

