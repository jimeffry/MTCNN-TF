import numpy as np
def py_nms(dets, thresh, mode="Union"):
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
    #order = scores.argsort()
    #print("order, ",order)

    keep = []
    while order.size > 0:
        i = order[0]
        #print("i",i)
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
        #get the opsite of the condition
        inds = np.where(ovr <= thresh)[0]
        #print("inds ",inds+1)
        # inds inlcude the first one : 0, inds+1 is keeping the <thresh;
        # because areas[order[1:]], so the lenth of order[1:] is less one than orignal order. so inds should plus 1
        order = order[inds+1]
    return keep


def py_nms_(boxes, threshold, mode='Union'):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = []
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == 'Minimum':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick

def py_nms__(boxes,overlapthresh,mode='Union'):
    if boxes.shape[0] == 0:
        return np.array([])
#    if boxes.dtype.kind=="i":
#        boxes=boxes.astype("float")
    pick=[]
    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    score=boxes[:,4]
    area=(x2-x1+1)*(y2-y1+1)
    idx=np.argsort(score)
    while len(idx)>0:
        last=len(idx)-1
        i=idx[last]
        pick.append(i)
        xx1=np.maximum(x1[i],x1[idx[:last]])
        yy1=np.maximum(y1[i],y1[idx[:last]])
        xx2=np.minimum(x2[i],x2[idx[:last]])
        yy2=np.minimum(y2[i],y2[idx[:last]])
        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)
        if mode =="Union":
            overlap=(w*h)/(area[idx[:last]] + area[idx[last]] - w*h)
        else:
            overlap=(w*h)/np.minimum(area[idx[:last]], area[idx[last]])
        idx=np.delete(idx,np.concatenate(([last],np.where(overlap >overlapthresh)[0])))
    return pick

if __name__ =='__main__':
    dets = np.array([[5,5,10,10,0.9],[7,7,20,20,0.7],[10,10,30,30,0.8],[40,40,50,50,0.3]])
    k = py_nms(dets,0.1)
    p = nms_(dets,0.1)
    print("array",dets[k])
    print("array",dets[p])
