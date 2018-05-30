#coding:utf-8
from mtcnn_model import O_Net,P_Net,R_Net
from train import train
import argparse

def args():
    parser = argparse.ArgumentParser(description='train Mtcnn')
    parser.add_argument('--train_net',type=str,default="PNet",\
                        help="begin to train which net -PNet RNet ONet")
    parser.add_argument('--train_data_set',type=str,default='WiderFace',\
                        help="begin to train which net -PNet RNet ONet")
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
        net_factory = P_Net
    elif train_net == "RNet":
        net_factory = R_Net
    else:
        net_factory = O_Net
    train(net_factory, prefix,load_epoch, end_epoch, base_dir, display=display, base_lr=lr)

if __name__ == '__main__':
    #base_dir = '../prepare_data/imglists/ONet'
    parm = args()
    train_net = parm.train_net
    train_data_set = parm.train_data_set
    base_dir = '../prepare_data/%s_imglists/%s' % (train_data_set,train_net)
    #model_name = 'MTCNN_model'
    model_name = 'MTCNN_%s_model' % train_data_set
    model_path = '../data/%s/ONet_landmark/%s' % (model_name,train_net)
    prefix = model_path
    end_epoch = 200000
    display = 1000
    lr = 0.01
    load_epoch = 0
    train_ONet(base_dir, prefix,load_epoch, end_epoch, display, lr,train_net)
