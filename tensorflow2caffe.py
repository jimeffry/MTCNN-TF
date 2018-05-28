#author: lxy
#time: 2018.4.3/ 11:30
#tool: python3
#version: 0.1
#project: face-detect
#modify:
#########################################
import numpy as np 
import sys
import os
sys.path.append("/home/lxy/caffe/python")
os.environ['GLOG_minloglevel'] = '2'
import caffe
#import caffe.net as caffe_net
import tensorflow as tf
#from caffe import caffe_net 
from train_models.MTCNN_config import config
from train_models.mtcnn_model import P_Net,R_Net,O_Net
import argparse

GLOG_minloglevel=1

def convert_filter(numpy_filter_weight):
    return np.transpose(numpy_filter_weight,(3,2,1,0))

def convert_fc(numpy_fc_weight):
    return np.transpose(numpy_fc_weight,(1,0))

def get_tf(model_file,train_net):
    #sess = tf.Session()
    graph = tf.Graph()
    test_fg = config.train_face
    key_list = []
    var_dic = dict()
    with graph.as_default():
        if str(train_net) == "PNet":
            image_op = tf.placeholder(tf.float32, shape=[1, 12, 12, 3], name='input')
            net_factory = P_Net
        elif str(train_net) == 'RNet':
            image_op = tf.placeholder(tf.float32, shape=[1, 24, 24, 3], name='input')
            net_factory = R_Net
        elif str(train_net) == 'ONet':
            image_op = tf.placeholder(tf.float32, shape=[1, 48, 48, 3], name='input')
            net_factory = O_Net
        #figure out landmark
        if test_fg:
            cls_prob, bbox_pred, landmark_pred = net_factory(image_op, training=False)
        else:
            cls_prob, bbox_pred = net_factory(image_op, training=False)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        #new_saver = tf.train.import_meta_graph(model_file+".meta")
        saver = tf.train.Saver(max_to_keep=0)
        module_ = saver.restore(sess,model_file)
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        conv1 = all_vars[0]
        bias1 = all_vars[1]
        #for var_ in all_vars:
            #print(var_)
        for v_name in tf.global_variables():
            print("name : ",v_name.name[:-2],v_name.shape) 
            key_list.append(v_name.name[:-2])
            #print(tf.get_variable_scope())
        conv_w1, bias_1 , vas= sess.run([conv1,bias1,all_vars])
        print("conv ",np.shape(conv_w1))
        print("bn ",np.shape(bias_1))
        print(len(vas))
        for i in range(len(vas)):
            cur_name = key_list[i]
            cur_var = vas[i]
            if "weight" in cur_name and "fc" not in cur_name:
                cur_var = convert_filter(cur_var)
                var_dic[cur_name] = cur_var
            elif "fc" in cur_name and "weight" in cur_name:
                cur_var = convert_fc(cur_var)
                var_dic[cur_name] = cur_var
            else:
                var_dic[cur_name]= cur_var
        print("dic0: ",np.shape(var_dic['conv1/weights']))
    return var_dic

def get_caffe(var_dic,protxt_path,caffe_model_path):
    net = caffe.Net(protxt_path, caffe.TEST)
    '''
    for layer_name, blob in net.blobs.iteritems():  
        print (layer_name + '\t' + str(blob.data.shape) )
    '''
    print("begin to out param :")
    #print(net.params('conv1', 1).get_data().shape)
    for layer_name, param in net.params.iteritems():  
        print(layer_name)    
        if 'conv' in layer_name:
            #print(net.params('conv1', 1).get_data())
            print (layer_name,param[0].data.shape,np.shape(var_dic[layer_name+'/weights']))
            print(layer_name+'/weights')
            param[0].data[:,:,:,:] = var_dic[layer_name+'/weights']
            #param[0].set_data(var_dic[layer_name+'/weights'])   
            param[1].data[:] = var_dic[layer_name+'/biases']
        elif "fc1" in layer_name:
            print (layer_name,param[0].data.shape,np.shape(var_dic[layer_name+'/weights']))
            param[0].data[:,:] = var_dic[layer_name+'/weights']   
            param[1].data[:] = var_dic[layer_name+'/biases']
        elif "cls" in layer_name:
            print (layer_name,param[0].data.shape,np.shape(var_dic[layer_name+'/weights']))
            param[0].data[:,:] = var_dic[layer_name+'/weights']   
            param[1].data[:] = var_dic[layer_name+'/biases']
        elif "bbox" in layer_name:
            print (layer_name,param[0].data.shape,np.shape(var_dic[layer_name+'/weights']))
            param[0].data[:,:] = var_dic[layer_name+'/weights']   
            param[1].data[:] = var_dic[layer_name+'/biases']
        elif "landmark" in layer_name:
            print (layer_name,param[0].data.shape,np.shape(var_dic[layer_name+'/weights']))
            param[0].data[:,:] = var_dic[layer_name+'/weights']   
            param[1].data[:] = var_dic[layer_name+'/biases']
    net.save(caffe_model_path) 
      

#def gen_caffe():
    #caffe_net.Caffemodel('')

def load_caffe(pro_path,model_path):
    net = caffe.Net(protxt_path,model_path, caffe.TEST)
    for layer_name, blob in net.blobs.iteritems():  
        print (layer_name + '\t' + str(blob.data.shape) )

def args():
    parser = argparse.ArgumentParser(description="tensorflow to caffe")
    parser.add_argument('--test_net',type=str,default="PNet",help="which net to convert")
    parser.add_argument('--test_load',type=bool,default=False,help="whether convert is successful")
    return parser.parse_args()

if __name__=='__main__':
    #os.environ(::google::InitGoogleLogging(" "))
    arg = args()
    test_net = arg.test_net
    test_fg = arg.test_load
    if test_net=="PNet":
        #path_ = './data/MTCNN_bright_model/PNet_landmark/PNet-205'
        #path_= './data/MTCNN_model/PNet_landmark/v1_trained/PNet-32'
        path_= './data/MTCNN_model/PNet_landmark/resaved/PNet-5relu'
        protxt_path = './caffe/PNet.prototxt'
        caffe_model_path = './caffe/PNet.caffemodel'
        net_factory = "PNet"
    elif test_net=="RNet":
        #path_= './data/MTCNN_model/RNet_landmark/v1_trained/RNet-4400'
        path_= './data/MTCNN_model/RNet_landmark/resaved/RNet-40relu'
        protxt_path = './caffe/RNet.prototxt'
        caffe_model_path = './caffe/RNet.caffemodel'
        net_factory = 'RNet'
    elif test_net=="ONet":
        #path_= './data/MTCNN_model/ONet_landmark/v1_trained/ONet-25'
        path_= './data/MTCNN_model/ONet_landmark/resaved/ONet-60relu'
        protxt_path = './caffe/ONet.prototxt'
        caffe_model_path = './caffe/ONet.caffemodel'
        net_factory = 'ONet'
    if test_fg:
        load_caffe(path_,caffe_model_path)
    else:
        parameters = get_tf(path_,net_factory)
        get_caffe(parameters,protxt_path,caffe_model_path)