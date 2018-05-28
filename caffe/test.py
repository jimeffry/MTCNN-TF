import sys
sys.path.append('/home/lxy/caffe/python')
import caffe
import cv2
import numpy as np
import time

class MtcnnDetector(object):
    def __init__(self,min_face,score_thresh,threshold=[0.5,0.7,0.6]):
        deploy = 'PNet.prototxt'
        caffemodel = 'PNet.caffemodel'
        self.net_12 = caffe.Net(deploy,caffemodel,caffe.TEST)
        deploy = 'RNet.prototxt'
        caffemodel = 'RNet.caffemodel'
        self.net_24 = caffe.Net(deploy,caffemodel,caffe.TEST)
        deploy = 'ONet.prototxt'
        caffemodel = 'ONet.caffemodel'
        self.net_48 = caffe.Net(deploy,caffemodel,caffe.TEST)
        self.threshold = threshold
        self.score_thresh = score_thresh
        self.min_face = min_face
        self.pnet_detector = 1
        self.rnet_detector = 1
        self.onet_detector = 1
        self.scale_factor = 0.79

    def generate_bbox(self,cls_map, reg, scale, threshold):
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
        t_index = np.where(cls_map >= threshold)
        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        #offset
        dx1, dy1, dx2, dy2 = [reg[i,t_index[0], t_index[1]] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])
        return boundingbox.T

    def py_nms(self,dets, threshold, mode="Union"):
        """
        greedily select boxes with high confidence
        keep boxes overlap <= thresh
        rule out overlap > thresh
        :param dets: [[x1, y1, x2, y2 score]]
        :param thresh: retain overlap <= thresh
        :return: indexes to keep
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if mode == "Union":
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == "Minimum":
                ovr = inter / np.minimum(areas[i], areas[order[1:]])
            #keep
            #print("len over ",len(ovr))
            inds = np.where(ovr <= (1-threshold))[0]
            #print("inds ",inds)
            order = order[inds + 1]
        return keep

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

    def calibrate_box(self, bbox, reg):
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

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

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
        keep = np.where(bboxes[:,0]<w-1)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,1]<h-1)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,2]>0)
        bboxes = bboxes[keep]
        keep = np.where(bboxes[:,3]>0)
        bboxes = bboxes[keep]
        #print("pad ",bboxes.shape)
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

    def processed_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def detect_pnet(self,img):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        caffe_img = (img-127.5)/128
        origin_h,origin_w,ch = caffe_img.shape
        out = []
        cur_scale = 12.0/self.min_face
        img_resized = self.processed_image(img,cur_scale)
        cur_h,cur_w,_ = img_resized.shape
        cnt_img = 0
        scales = []
        while min(cur_h,cur_w) >12:
            #img_resized = img
            #cur_scale = 1.0
            #cur_h,cur_w,_ = img_resized.shape
            #if True:
            hs = int(cur_h)
            ws = int(cur_w)
            #scale_img = np.swapaxes(img_resized, 0, 2)
            scale_img = np.transpose(img_resized,(2,0,1))
            self.net_12.blobs['data'].reshape(1,3,hs,ws)
            self.net_12.blobs['data'].data[...]=scale_img
            out_ = self.net_12.forward()
            out.append(out_)
            scales.append(cur_scale)
            cnt_img +=1
            cur_scale*=self.scale_factor
            img_resized = self.processed_image(img,cur_scale)
            cur_h,cur_w,_ = img_resized.shape
        rectangles = []
        for i in range(cnt_img):    
            cls_prob = out[i]['prob1'][0][1]
            roi      = out[i]['conv4_2'][0]
            rectangle = self.generate_bbox(cls_prob, roi, scales[i], self.score_thresh[0])
            if rectangle.size==0:
                continue
            keep = self.py_nms(rectangle[:,:5],self.threshold[0])
            rectangles.append(rectangle[keep])
        if len(rectangles)==0:
            return None
        rectangles = np.vstack(rectangles)
        keep = self.py_nms(rectangles,self.threshold[0])
        all_boxes = rectangles[keep]
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        #refine the boxes
        #print('pnet box ',np.shape(all_boxes))
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                                all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                                all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                                all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                                all_boxes[:, 4]])
        boxes_c = boxes_c.T
        return all_boxes,boxes_c

    def detect_rnet(self,img,bboxes):
        h, w, c = img.shape
        bboxes = self.convert_to_square(bboxes)
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(bboxes, w, h)
        num_boxes = bboxes.shape[0]
        self.net_24.blobs['data'].reshape(num_boxes,3,24,24)
        print("image shape ",h,w)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            cropped_img = np.zeros((24, 24, 3), dtype=np.float32)
            #print("box: ", bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3])
            #print("rnet: ",y[i],ey[i],x[i],ex[i])
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_img = (cv2.resize(tmp, (24, 24))-127.5) / 128
            #scale_img = np.swapaxes(cropped_img, 0, 2)
            scale_img = np.transpose(cropped_img,(2,0,1))
            self.net_24.blobs['data'].data[i] =scale_img
        #cls_scores : num_data*2
        #reg: num_data*4
        #landmark: num_data*10
        out = self.net_24.forward()
        cls_prob = out['prob1']
        roi_prob = out['bbox_fc']
        landmark = out['landmark_fc']
        cls_scores = cls_prob[:,1]
        keep_inds = np.where(cls_scores > self.score_thresh[1])[0]
        if len(keep_inds) > 0:
            boxes_new = bboxes[keep_inds]
            boxes_new[:, 4] = cls_scores[keep_inds]
            reg = roi_prob[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None        
        #width
        w = boxes_new[:,2] - boxes_new[:,0] + 1
        #height
        h = boxes_new[:,3] - boxes_new[:,1] + 1
        landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes_new[:,0],(5,1)) - 1).T
        landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes_new[:,1],(5,1)) - 1).T 
        keep = self.py_nms(boxes_new, self.threshold[1])
        boxes_c = boxes_new[keep]
        boxes_c = self.calibrate_box(boxes_c, reg[keep])
        landmark = landmark[keep]
        return boxes_c,landmark

    def detect_onet(self,img,bboxes):
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
        h, w, c = img.shape
        bboxes = self.convert_to_square(bboxes)
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(bboxes, w, h)
        num_boxes = bboxes.shape[0]
        self.net_48.blobs['data'].reshape(num_boxes,3,48,48)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            cropped_img = np.zeros((48, 48, 3), dtype=np.float32)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_img = (cv2.resize(tmp, (48, 48))-127.5) / 128
            scale_img = np.transpose(cropped_img,(2,0,1))
            self.net_48.blobs['data'].data[i] = scale_img
        out = self.net_48.forward()
        cls_prob = out['prob1']
        roi_prob = out['bbox_fc']
        landmark = out['landmark_fc']
        #prob belongs to face
        cls_scores = cls_prob[:,1]        
        keep_inds = np.where(cls_scores > self.score_thresh[2])[0]        
        if len(keep_inds) > 0:
            #pickout filtered box
            boxes_new = bboxes[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = roi_prob[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None
        #width
        w = boxes_new[:,2] - boxes_new[:,0] + 1
        #height
        h = boxes_new[:,3] - boxes_new[:,1] + 1
        if self.train_face:
            landmark[:,0::2] = (np.tile(w,(5,1)) * landmark[:,0::2].T + np.tile(boxes_new[:,0],(5,1)) - 1).T
            landmark[:,1::2] = (np.tile(h,(5,1)) * landmark[:,1::2].T + np.tile(boxes_new[:,1],(5,1)) - 1).T        
        boxes_c = self.calibrate_box(boxes_new, reg)         
        keep = self.py_nms(boxes_c,self.threshold[2], "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes_c,landmark

    def detect(self,img):
        """Detect face over image
        """
        boxes = None
        t = time.time()
        # pnet
        t1 = 0
        if self.pnet_detector:
            all_box,boxes_c = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([])
            t1 = time.time() - t
            t = time.time()
            print("Pnet out ",boxes_c.shape)
        # rnet 
        for i in range(10):
            print("box_c ",map(int,boxes_c[i]))
            print("box ",map(int,all_box[i])) 
        t2 = 0
        if self.rnet_detector:
            boxes_c,landmark = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
            t2 = time.time() - t
            t = time.time()
        # onet 
        t3 = 0
        if self.onet_detector:
            boxes_c,landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]),np.array([])
            t3 = time.time() - t
            t = time.time()
            print(
                "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,t3))
        return boxes_c,landmark

def detectFace(self,img,threshold,min_face):
        net_24.blobs['data'].reshape(len(boxes_c),3,24,24)
        crop_number = 0
        for rectangle in boxes_c:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(24,24))
            scale_img = np.swapaxes(scale_img, 0, 2)
            net_24.blobs['data'].data[crop_number] =scale_img 
            crop_number += 1
        out = net_24.forward()
        cls_prob = out['prob1']
        roi_prob = out['bbox_fc']
        rectangles = tools.filter_face_24net(cls_prob,roi_prob,rectangles,origin_w,origin_h,threshold[1])
        
        if len(rectangles)==0:
            return rectangles
        net_48.blobs['data'].reshape(len(rectangles),3,48,48)
        crop_number = 0
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img,(48,48))
            scale_img = np.swapaxes(scale_img, 0, 2)
            net_48.blobs['data'].data[crop_number] =scale_img 
            crop_number += 1
        out = net_48.forward()
        cls_prob = out['prob1']
        roi_prob = out['bbox_fc']
        pts_prob = out['landmark_fc']
        rectangles = tools.filter_face_48net(cls_prob,roi_prob,pts_prob,rectangles,origin_w,origin_h,threshold[2])
        return rectangles

