import tensorflow as tf
import numpy as np
import os
#from print_ckpt import print_ckpt
from tensorflow.contrib import slim
import sys
sys.path.append("../")
from train_models.MTCNN_config import config

class Detector(object):
    #net_factory:rnet or onet
    #datasize:24 or 48
    def __init__(self, net_factory, data_size, batch_size, model_path):
        graph = tf.Graph()
        self.test_fg = config.train_face
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input')
            #figure out landmark
            if self.test_fg:
                self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
                if self.test_fg == 100:
                    self.landmark_pred = tf.identity(self.landmark_pred,name='output')
            else:
                self.cls_prob, self.bbox_pred = net_factory(self.image_op, training=False)
                #self.cls_prob = tf.identity(self.cls_prob,name='cls_out')
                #self.bbox_pred = tf.identity(self.bbox_pred,name='bbox_out')
                #self.landmark_pred = tf.identity(self.landmark_pred,name='out')
                #self.output_op = tf.concat([self.cls_prob, self.bbox_pred], 1)
                #self.net_out = slim.flatten(self.output_op,scope='flatten_1')
                #self.out_put = tf.identity(self.net_out,name='output')

            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            net_name = model_path.split('/')[-1]
            print("net name is ",net_name)
            if self.test_fg==100:
                logs_dir = "../logs/%s" %(net_name)
                summary_op = tf.summary.merge_all()
                if os.path.exists(logs_dir) == False:
                    os.mkdir(logs_dir)
                writer = tf.summary.FileWriter(logs_dir,self.sess.graph)
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print ("model_dict is ",model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            #assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path)
            if self.test_fg==100:
                saver.save(self.sess,model_dict+'/resaved/'+net_name+'relu')
            #print_ckpt('./checkpoint')


        self.data_size = data_size
        self.batch_size = batch_size
    #rnet and onet minibatch(test)
    def predict(self, databatch):
        # access data
        # databatch: N x 3 x data_size x data_size
        scores = []
        batch_size = self.batch_size
        minibatch = []
        cur = 0
        #num of all_data
        n = databatch.shape[0]
        while cur < n:
            #split mini-batch
            minibatch.append(databatch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        #every batch prediction result
        cls_prob_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx, data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #the last batch
            if m < batch_size:
                keep_inds = np.arange(m)
                #gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            #cls_prob batch*2
            #bbox_pred batch*4
            if self.test_fg:
                cls_prob, bbox_pred,landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred,self.landmark_pred], feed_dict={self.image_op: data})
                #num_batch * batch_size*10
                landmark_pred_list.append(landmark_pred[:real_size])
            else:
                cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred], feed_dict={self.image_op: data})
            #num_batch * batch_size *2
            cls_prob_list.append(cls_prob[:real_size])
            #num_batch * batch_size *4
            bbox_pred_list.append(bbox_pred[:real_size])                   
            #num_of_data*2,num_of_data*4,num_of_data*10
        if config.Debug:
            print("detect shape cls box landmark : ",np.shape(cls_prob_list),np.shape(bbox_pred_list),np.shape(landmark_pred_list))
        if self.test_fg:
            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)
        else:
            return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0)
