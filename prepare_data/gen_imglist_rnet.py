import numpy as np
import numpy.random as npr
import os
test=0
data_dir = '.'
#anno_file = os.path.join(data_dir, "anno.txt")

size = 24

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"

with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
    part = f.readlines()

with open(os.path.join(data_dir, '%s/landmark_%s_aug.txt' % (size, size)), 'r') as f:
    landmark = f.readlines()
  
dir_path = os.path.join(data_dir, 'imglists',"RNet")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
#write all data
with open(os.path.join(dir_path, "train_caffe_%s_landmark.txt" % (net)), "w") as f:
    print len(neg)
    print len(pos)
    print len(part)
    print len(landmark)
    str_po = ' -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 '
    str_neg = ' -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 '
    str_mark = ' -1 -1 -1 -1 -1 '
    if test :
        cnt = 5
        cnt_neg = 5
        cnt_pa = 5
        cnt_ma = 5
    else:
        cnt = len(pos)
        cnt_neg = len(neg)
        cnt_pa = len(part)
        cnt_ma = len(landmark)
    for i in np.arange(cnt):
        line_pos = pos[i].strip()
        f.write(line_pos+str_po+'\n')
    for i in np.arange(cnt_neg):
        line_pos = neg[i].strip()
        f.write(line_pos+str_neg+'\n')
    for i in np.arange(cnt_pa):
        line_pos = part[i].strip().split()
        line_pos[1] = str(0)
        f.write(' '.join(line_pos) +str_po+'\n')
    for i in np.arange(cnt_ma):
        line_pos = landmark[i].strip().split()
        f.write(line_pos[0]+str_mark+ ' '.join(line_pos[2:]) +'\n')
