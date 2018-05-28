#!/usr/bin/bash
#pnet
#python gen_12net_data.py
#python gen_imglist_pnet.py   --pos_txt=WiderFace_pos12.txt --neg_txt=WiderFace_neg12.txt --part_txt=WiderFace_part12.txt  --out_dir=./WiderFace_imglists --base_num=360000
python gen_PNet_tfrecords.py --data_out_dir=WiderFace_imglists/PNet
#rnet
#python gen_hard_example.py --test_mode=PNet
python gen_imglist_onet.py  --data_dir=./24 --size=24  --pos_txt=WiderFace_pos24.txt --neg_txt=WiderFace_neg24.txt --part_txt=WiderFace_part24.txt --landmark=True --out_dir=./WiderFace_imglists --base_num=200000
python gen_RNet_tfrecords.py --read_type=pos --anno_file=24/WiderFace_pos24.txt --gen_net=24 --data_out_dir=WiderFace_imglists/RNet
python gen_RNet_tfrecords.py --read_type=part --anno_file=24/WiderFace_part24.txt --gen_net=24 --data_out_dir=WiderFace_imglists/RNet
python gen_RNet_tfrecords.py --read_type=neg --anno_file=24/WiderFace_neg24.txt --gen_net=24 --data_out_dir=WiderFace_imglists/RNet

