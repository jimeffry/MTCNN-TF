#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 256
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [640,1280,25600,51200]
# 5:test relu, 100: generate for MVtensor
config.train_face = 5
config.r_out = 1
config.P_Num = 500
config.rnet_wide =1
config.o_out =0
config.Debug =0
config.p_landmark=0
#1:train, 0:test
config.train_mode=0