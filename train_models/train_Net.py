#coding:utf-8
from mtcnn_model import O_Net,P_Net,R_Net,P_Net_W
from train import train
import argparse
import os
from MTCNN_config import config


def args():
    parser = argparse.ArgumentParser(description='train Mtcnn')
    parser.add_argument('--train_net',type=str,default="PNet",\
                        help="begin to train which net -PNet RNet ONet")
    parser.add_argument('--train_data_set',type=str,default='widerface',\
                        help="begin to train which net -PNet RNet ONet")
    parser.add_argument('--load_epoch',type=int,default=0,\
                        help="load the pretrained model")
    parser.add_argument('--dispaly_num',type=int,default=1000,\
                        help="every num to print the info")
    parser.add_argument('--end_epoch',type=int,default=500000,\
                        help="the epoch num to train")
    parser.add_argument('--learn_rate',type=float,default=0.01,\
                        help="the training learning rate begin")
    parser.add_argument('--gpu',type=str,default='0',\
                        help="which gpu to run")   
    return parser.parse_args()

def train_ONet(base_dir, prefix, load_epoch,end_epoch, display, lr,train_net):
    """
    train PNet
    :param dataset_dir: tfrecord path
    :param prefix:
    :param end_epoch:
    :param display:
    :param lr:
    :return:
    """
    if train_net == "PNet":
        if config.train_face:
            net_factory = P_Net
        else:
            net_factory = P_Net_W
    elif train_net == "RNet":
        net_factory = R_Net
    else:
        net_factory = O_Net
    train(net_factory, prefix,load_epoch, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    #base_dir = '../prepare_data/imglists/ONet'
    parm = args()
    gpu_num = parm.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    train_net = parm.train_net
    train_data_set = parm.train_data_set
    base_dir = '../prepare_data/%s_imglists/%s' % (train_data_set,train_net)
    #model_name = 'MTCNN_model'
    model_name = 'MTCNN_%s_model' % train_data_set
    model_path = '../data/%s/%s_landmark/%s' % (model_name,train_net,train_net)
    prefix = model_path
    end_epoch = parm.end_epoch
    display = parm.dispaly_num
    lr = parm.learn_rate
    load_epoch = parm.load_epoch
    train_ONet(base_dir, prefix,load_epoch, end_epoch, display, lr,train_net)
