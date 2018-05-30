#coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from MTCNN_config import config
num_keep_radio = 0.7
#define prelu
test_fg = config.train_face
p_landmark = config.p_landmark
def prelu(inputs):
    #alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    alphas = 0.25
    pos = tf.nn.relu(inputs)
    if test_fg == 100 or test_fg==5:
        return pos
    else:
        #neg = alphas * (inputs-abs(inputs))*0.5
        neg = 0.25 * (inputs-abs(inputs))*0.5
        return pos +neg

def dense_to_one_hot(labels_dense,num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes
    #num_sample*num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
#cls_prob:batch*2
#label:batch

def cls_ohem(cls_prob, label):
    zeros = tf.zeros_like(label)
    ones_index = tf.ones_like(label)
    #label=-1 --> label=0 net_factory
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    #label=-2 --> label=1 : lxy
    #label_filter_invalid = tf.where(tf.equal(label,-1), zeros, label)
    #label_filter_invalid = tf.where(tf.equal(label,-2), ones_index, label)
    num_cls_prob = tf.size(cls_prob)
    # cls_prob_reshape shpae is [batch_size *2, -1]
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    # num_row=batch_size, so row is [ 0,2,4,..., 254], and row is negtive label,corresponding cls_prob[:,0]
    row = tf.range(num_row)*2
    #so indices is the net out:0,1 ,according to label select the ground or face
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    #label: 0 -->1, to calculate valid num
    #valid_inds = tf.where(label < zeros,zeros,ones)
    #lxy: -1 -->1, part should be neg
    valid_inds = tf.where(label_filter_invalid  <zeros, zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)
def bbox_ohem_smooth_L1_loss(bbox_pred,bbox_target,label):
    sigma = tf.constant(1.0)
    threshold = 1.0/(sigma**2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    abs_error = tf.abs(bbox_pred-bbox_target)
    loss_smaller = 0.5*((abs_error*sigma)**2)
    loss_larger = abs_error-0.5/(sigma**2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error<threshold,loss_smaller,loss_larger),axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    smooth_loss = smooth_loss*valid_inds
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)
def bbox_ohem_orginal(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    #pay attention :there is a bug!!!!
    valid_inds = tf.where(label!=zeros_index,tf.ones_like(label,dtype=tf.float32),zeros_index)
    #(batch,)
    square_error = tf.reduce_sum(tf.square(bbox_pred-bbox_target),axis=1)
    #keep_num scalar
    keep_num = tf.cast(tf.reduce_sum(valid_inds)*num_keep_radio,dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
#label=1 or label=-1 then do regression
def bbox_ohem(bbox_pred,bbox_target,label):
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    #valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    valid_inds = tf.where(tf.equal(label, 1),ones_index,zeros_index)
    #(batch,)
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    #keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #keep valid index square_error
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def landmark_ohem(landmark_pred,landmark_target,label):
    #keep label =-2  then do landmark detection
    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)
    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error*valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)

def cal_accuracy(cls_prob,label):
    pred = tf.argmax(cls_prob,axis=1)
    label_int = tf.cast(label,tf.int64)
    cond = tf.where(tf.greater_equal(label_int,0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int,picked)
    pred_picked = tf.gather(pred,picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked,pred_picked),tf.float32))
    return accuracy_op
#construct Pnet
#label:batch

def print_shape(net,name,conv_num):
    print("the net {} in {}  shape is {} ".format(name,net,[conv_num.get_shape()]))

def P_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        #print ("Pnet input shape",inputs.get_shape())
        #net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        net = slim.conv2d(inputs, 8, 3, stride=1,scope='conv1')
        #print ("conv1 shape ",net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        #print ("pool1 shape ",net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        #print ("conv2 shape ",net.get_shape())
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        #print ("conv3 shape ",net.get_shape())
        #batch*H*W*2
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        #conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)

        #print ("cls shape ",conv4_1.get_shape())
        #batch*H*W*4
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        #print ("bbox shape ",bbox_pred.get_shape())
        #batch*H*W*10
        if p_landmark:
            landmark_pred = slim.conv2d(net,num_outputs=10,kernel_size=[1,1],stride=1,scope='conv4_3',activation_fn=None)
            #print ("landmark shape ",landmark_pred.get_shape())
        #cls_prob_original = conv4_1
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            #batch
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*10
            if p_landmark:
                landmark_pred = tf.squeeze(landmark_pred,[1,2],name="landmark_pred")
                landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            else:
                landmark_loss = 0
            accuracy = cal_accuracy(cls_prob,label)
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        #test
        else:
            #when test,batch_size = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_pred_test = tf.squeeze(bbox_pred,axis=0)
            if p_landmark:
                landmark_pred_test = tf.squeeze(landmark_pred,axis=0)
                return cls_pro_test,bbox_pred_test,landmark_pred_test
            else:           
                return cls_pro_test,bbox_pred_test
'''
def P_Net(inputs,label=None,bbox_target=None,training=True):
    #define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print ("Pnet input shape",inputs.get_shape())
        #net = slim.conv2d(inputs, 10, 3, stride=1,scope='conv1')
        net = slim.conv2d(inputs, 8, 3, stride=1,scope='conv1')
        print ("conv1 shape ",net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2,2], stride=2, scope='pool1', padding='SAME')
        print ("pool1 shape ",net.get_shape())
        net = slim.conv2d(net,num_outputs=16,kernel_size=[3,3],stride=1,scope='conv2')
        print ("conv2 shape ",net.get_shape())
        net = slim.conv2d(net,num_outputs=32,kernel_size=[3,3],stride=1,scope='conv3')
        print ("conv3 shape ",net.get_shape())
        #batch*H*W*2
        conv4_1 = slim.conv2d(net,num_outputs=2,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.softmax)
        #conv4_1 = slim.conv2d(net,num_outputs=1,kernel_size=[1,1],stride=1,scope='conv4_1',activation_fn=tf.nn.sigmoid)

        print ("cls shape ",conv4_1.get_shape())
        #batch*H*W*4
        bbox_pred = slim.conv2d(net,num_outputs=4,kernel_size=[1,1],stride=1,scope='conv4_2',activation_fn=None)
        print ("bbox shape ",bbox_pred.get_shape())
        #cls_prob_original = conv4_1
        #bbox_pred_original = bbox_pred
        if training:
            #batch*2
            cls_prob = tf.squeeze(conv4_1,[1,2],name='cls_prob')
            cls_loss = cls_ohem(cls_prob,label)
            #batch
            bbox_pred = tf.squeeze(bbox_pred,[1,2],name='bbox_pred')
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            #batch*10
            landmark_loss = 0
            accuracy = cal_accuracy(cls_prob,label)
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        #test
        else:
            #when test,batch_size = 1        
            return cls_pro_test,bbox_pred_test
'''

def R_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        #print_shape('RNet','input',inputs)
        #net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3,3], stride=1, scope="conv1")
        if config.rnet_wide:
            net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        else:
            net = slim.conv2d(inputs, num_outputs=16, kernel_size=[3,3], stride=1, scope="conv1")
        print_shape('RNet','conv1',net)
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print_shape('RNet','pool1',net)
        #net = slim.conv2d(net,num_outputs=48,kernel_size=[3,3],stride=1,scope="conv2")
        if config.rnet_wide:
            net = slim.conv2d(net, num_outputs=64, kernel_size=[3,3], stride=1, scope="conv2")
        else:
            net = slim.conv2d(net, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv2")
        print_shape('RNet','conv2',net)
        if config.rnet_wide:
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2",padding='SAME')
        else:
            net = slim.max_pool2d(net,kernel_size=[3,3],stride=2,scope="pool2")
        print_shape('RNet','pool2',net)
        if config.rnet_wide:
            net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
            print_shape('RNet','conv3',net)
            net = slim.conv2d(net,num_outputs=128,kernel_size=[3,3],stride=1,scope="conv4")
            print_shape('RNet','conv4',net)
        else:
            net = slim.conv2d(net,num_outputs=64,kernel_size=[2,2],stride=1,scope="conv3")
            print_shape('RNet','conv3',net)
        fc_flatten = slim.flatten(net)
        print_shape('RNet','flatten',fc_flatten)
        if config.rnet_wide:
            fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        else:
            fc1 = slim.fully_connected(fc_flatten, num_outputs=128,scope="fc1", activation_fn=prelu)
        print_shape('RNet','fc1',fc1)
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print_shape('RNet','cls_fc',cls_prob)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print_shape('RNet','bbox_fc',bbox_pred)
        #batch*10
        if  test_fg :
            landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
            print_shape('RNet','landmark_fc',landmark_pred)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            if  test_fg :
                landmark_loss = landmark_ohem(landmark_pred,landmark_target,label)
            else:
                landmark_loss = 0
            #landmark_loss = 0
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            if test_fg:
                return cls_prob,bbox_pred,landmark_pred
            else:
                return cls_prob,bbox_pred
            #return cls_prob,bbox_pred

def O_Net(inputs,label=None,bbox_target=None,landmark_target=None,training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn = prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        #print_shape('ONet','input',inputs)
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3,3], stride=1, scope="conv1")
        print_shape('ONet','conv1',net)
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print_shape('ONet','pool1',net)
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv2")
        print_shape('ONet','conv2',net)
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print_shape('ONet','pool2',net)
        net = slim.conv2d(net,num_outputs=64,kernel_size=[3,3],stride=1,scope="conv3")
        print_shape('ONet','conv3',net)
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print_shape('ONet','pool3',net)
        net = slim.conv2d(net,num_outputs=128,kernel_size=[2,2],stride=1,scope="conv4")
        print_shape('ONet','conv4',net)
        fc_flatten = slim.flatten(net)
        print_shape('ONet','flatten',fc_flatten)
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256,scope="fc1", activation_fn=prelu)
        print_shape('RNet','fc1',fc1)
        #batch*2
        cls_prob = slim.fully_connected(fc1,num_outputs=2,scope="cls_fc",activation_fn=tf.nn.softmax)
        print_shape('ONet','cls_fc',cls_prob)
        #batch*4
        bbox_pred = slim.fully_connected(fc1,num_outputs=4,scope="bbox_fc",activation_fn=None)
        print_shape('ONet','bbox_fc',bbox_pred)
        #batch*10
        if  test_fg:
            landmark_pred = slim.fully_connected(fc1,num_outputs=10,scope="landmark_fc",activation_fn=None)
            print_shape('RNet','landmark_fc',landmark_pred)
        #train
        if training:
            cls_loss = cls_ohem(cls_prob,label)
            bbox_loss = bbox_ohem(bbox_pred,bbox_target,label)
            accuracy = cal_accuracy(cls_prob,label)
            if  test_fg:
                landmark_loss = landmark_ohem(landmark_pred, landmark_target,label)
            else:
                landmark_loss = 0
            #landmark_loss = 0
            #L2_loss = tf.add_n(slim.losses.get_regularization_losses())
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss,bbox_loss,landmark_loss,L2_loss,accuracy
        else:
            if  test_fg:
                return cls_prob,bbox_pred,landmark_pred
            else:
                return cls_prob,bbox_pred
