!#/usr/bin/bash
#################
#caltech dataset
#Pnet
python gen_12net_data.py --anno_file=train_bright.txt --img_dir=black_bright --pos_save_dir=12/bright_positive  --part_save_dir=12/bright_part --neg_save_dir=12/bright_negative --pos_txt=bright_pos12.txt --neg_txt=bright_neg12.txt --part_txt=bright_part12.txt
python gen_imglist_pnet.py  --pos_txt=bright_pos12.txt --neg_txt=bright_neg12.txt --part_txt=bright_part12.txt
python gen_PNet_tfrecords.py
python3 train_Net.py --train_net=PNet --train_data_set=bright --load_epoch=80
#Rnet
python gen_hard_example.py --test_mode=PNet --min_face=24 --anno_file=./train_bright.txt  --img_saved_dir=./black_bright/  --train_data_set=bright
python gen_imglist_pnet.py  --data_dir=./24 --size=24  --pos_txt=bright_pos24.txt --neg_txt=bright_neg24.txt --part_txt=bright_part24.txt
python gen_RNet_tfrecords.py --read_type=pos --anno_file=24/bright_pos24.txt
python gen_RNet_tfrecords.py --read_type=part --anno_file=24/bright_part24.txt
python gen_RNet_tfrecords.py --read_type=neg --anno_file=24/bright_neg24.txt
python gen_RNet_tfrecords.py --read_type=landmark
python3 train_Net.py --train_net=RNet --train_data_set=bright --load_epoch=360
#Onet
python gen_hard_example.py --test_mode=RNet --min_face=24 --anno_file=./train_bright.txt --pos_txt=pos48.txt --neg_txt=neg48.txt --part_txt=part48.txt --img_saved_dir=./black_bright/   --train_data_set=bright
python gen_imglist_pnet.py  --data_dir=./48 --size=48  --pos_txt=bright_pos48.txt --neg_txt=bright_neg48.txt --part_txt=bright_part48.txt
python gen_RNet_tfrecords.py --read_type=pos --anno_file=48/bright_pos48.txt --gen_net=48 --data_out_dir=bright_imglists/ONet
python gen_RNet_tfrecords.py --read_type=part --anno_file=48/bright_part48.txt --gen_net=48 --data_out_dir=bright_imglists/ONet
python gen_RNet_tfrecords.py --read_type=neg --anno_file=48/bright_neg48.txt --gen_net=48 --data_out_dir=bright_imglists/ONet
python gen_RNet_tfrecords.py --read_type=landmark --gen_net=48 --data_out_dir=bright_imglists/ONet
python3 train_Net.py --train_net=ONet --train_data_set=bright --load_epoch=360

#############################################################
#Pnet
python gen_caltech_data.py --anno_file=train_caltech.txt --img_dir=caltech_images --pos_save_dir=12/caltech_positive  --part_save_dir=12/caltech_part --neg_save_dir=12/caltech_negative --pos_txt=caltech_pos12.txt --neg_txt=caltech_neg12.txt --part_txt=caltech_part12.txt
python gen_imglist_pnet.py  --data_dir=./12 --size=12  --pos_txt=caltech_pos12.txt --neg_txt=caltech_neg12.txt --part_txt=caltech_part12.txt --out_dir=./caltech_imglists
python gen_PNet_tfrecords.py --data_out_dir=caltech_imglists/PNet
python3 train_Net.py --train_net=PNet --train_data_set=caltech --load_epoch=80
#Rnet
python gen_hard_example.py --test_mode=PNet --min_face=24 --anno_file=./train_caltech.txt  --img_saved_dir=./caltech_images/  --train_data_set=caltech
python gen_imglist_pnet.py  --data_dir=./24 --size=24  --pos_txt=caltech_pos24.txt --neg_txt=caltech_neg24.txt --part_txt=caltech_part24.txt --out_dir=./caltech_imglists
python gen_RNet_tfrecords.py --read_type=pos --anno_file=24/caltech_pos24.txt --gen_net=24 --data_out_dir=caltech_imglists/RNet
python gen_RNet_tfrecords.py --read_type=part --anno_file=24/caltech_part24.txt --gen_net=24 --data_out_dir=caltech_imglists/RNet
python gen_RNet_tfrecords.py --read_type=neg --anno_file=24/caltech_neg24.txt --gen_net=24 --data_out_dir=caltech_imglists/RNet
python3 train_Net.py --train_net=RNet --train_data_set=caltech --load_epoch=360
#Onet
python gen_hard_example.py --test_mode=RNet --min_face=24 --anno_file=./train_caltech.txt  --img_saved_dir=./caltech_images/  --train_data_set=caltech --pos_txt=pos48.txt --neg_txt=neg48.txt --part_txt=part48.txt
(python gen_hard_example.py --test_mode=RNet --min_face=24 --anno_file=./train_caltech.txt  --img_saved_dir=./caltech_images/  --train_data_set=caltech --pos_txt=pos48.txt --neg_txt=neg48.txt --part_txt=part48.txt --ignore_det=True)
python gen_imglist_pnet.py  --data_dir=./48 --size=48  --pos_txt=caltech_pos48.txt --neg_txt=caltech_neg48.txt --part_txt=caltech_part48.txt --out_dir=./caltech_imglists
python gen_RNet_tfrecords.py --read_type=pos --anno_file=48/caltech_pos48.txt --gen_net=48 --data_out_dir=caltech_imglists/ONet
python gen_RNet_tfrecords.py --read_type=neg --anno_file=48/caltech_neg48.txt --gen_net=48 --data_out_dir=caltech_imglists/ONet
python gen_RNet_tfrecords.py --read_type=part --anno_file=48/caltech_part48.txt --gen_net=48 --data_out_dir=caltech_imglists/ONet
python3 train_Net.py --train_net=ONet --train_data_set=caltech --load_epoch=600
###################################################################
#Voc dataset
#Pnet
python gen_caltech_data.py --anno_file=train_voc.txt --img_dir=voc_images --pos_save_dir=12/voc_positive  --part_save_dir=12/voc_part --neg_save_dir=12/voc_negative --pos_txt=voc_pos12.txt --neg_txt=voc_neg12.txt --part_txt=voc_part12.txt
python gen_imglist_pnet.py  --data_dir=./12 --size=12  --pos_txt=voc_pos12.txt --neg_txt=voc_neg12.txt --part_txt=voc_part12.txt --out_dir=./voc_imglists
python gen_PNet_tfrecords.py --data_out_dir=voc_imglists/PNet
 python3 train_Net.py --train_net=PNet --train_data_set=voc --load_epoch=700 --learn_rate=0.001
#Rnet
python gen_RNet_hard_example.py --test_mode=PNet --min_face=24 --anno_file=./train_voc.txt  --img_saved_dir=./voc_images/  --train_data_set=voc
python gen_imglist_pnet.py  --data_dir=./24 --size=24  --pos_txt=voc_pos24.txt --neg_txt=voc_neg24.txt --part_txt=voc_part24.txt --out_dir=./voc_imglists
python gen_RNet_tfrecords.py --read_type=neg --anno_file=24/voc_neg24.txt --gen_net=24 --data_out_dir=voc_imglists/RNet
python gen_RNet_tfrecords.py --read_type=part --anno_file=24/voc_part24.txt --gen_net=24 --data_out_dir=voc_imglists/RNet
python3 train_Net.py --train_net=RNet --train_data_set=voc --load_epoch=4600
#Onet
python gen_RNet_hard_example.py --test_mode=RNet --min_face=24 --anno_file=./train_voc.txt  --img_saved_dir=./voc_images/  --train_data_set=voc --pos_txt=pos48.txt --neg_txt=neg48.txt --part_txt=part48.txt
python gen_imglist_pnet.py  --data_dir=./48 --size=48  --pos_txt=voc_pos48.txt --neg_txt=voc_neg48.txt --part_txt=voc_part48.txt --out_dir=./voc_imglists
python gen_RNet_tfrecords.py --read_type=neg --anno_file=48/voc_neg48.txt --gen_net=48 --data_out_dir=voc_imglists/ONet
python gen_RNet_tfrecords.py --read_type=part --anno_file=48/voc_part48.txt --gen_net=48 --data_out_dir=voc_imglists/ONet