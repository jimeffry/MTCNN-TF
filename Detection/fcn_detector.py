import numpy as np
import tensorflow as tf
import sys
sys.path.append("../")
from train_models.MTCNN_config import config
import os


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        graph = tf.Graph()
        self.train_face = config.train_face
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            if config.p_landmark:
                self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            else:
                self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #allow 
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            net_name = model_path.split('/')[-1]
            print("net name is ",net_name)
            if self.train_face==100:
                logs_dir = "../logs/%s" %(net_name)
                summary_op = tf.summary.merge_all()
                if os.path.exists(logs_dir) == False:
                    os.mkdir(logs_dir)
                writer = tf.summary.FileWriter(logs_dir,self.sess.graph)
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print("restore model path",model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print ("restore models' param")
            saver.restore(self.sess, model_path)
            if self.train_face==100:
                saver.save(self.sess,model_dict+'/resaved/'+net_name+'relu')
            '''
            logs_dir = "../logs/%s" %(net_factory)
            summary_op = tf.summary.merge_all()
            if os.path.exists(logs_dir) == False:
                os.mkdir(logs_dir)
            writer = tf.summary.FileWriter(logs_dir,self.sess.graph)
            #summary = self.sess.run()
            #writer.add_summary(summary,global_step=step)
            '''
    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred
