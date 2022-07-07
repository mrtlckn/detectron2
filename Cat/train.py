#>
# 
# Bu dosyayı calistirirken terminalden bu dosya konumuna git!!!

from detectron2.utils.logger import setup_logger

setup_logger()

#Register Dataset with detectron2
from detectron2.data.datasets import register_coco_instances

#Default trainer to train custom object detection and custom instance segmentation models
from detectron2.engine import DefaultTrainer

#libraries
import os
import pickle

from utils import *
############################################
#define the paths to configuration files and pre-trained model files
#OD
#config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#Instance Segmentation
config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


#pretrained model
#OD
#checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#Instance Segmentation
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
############################################


#where we want to save custom object detection model
#output_dir = "./output/object_detection"
output_dir = "./output/instance_segmentation"


#how many classes do we have
num_classes = 1

#where we want to train our model cuda or cpu
device = "cuda" #"cpu"

#Paths
train_dataset_name= "LP_train"
train_images_path = "train"
train_json_annot_path = "train.json"

test_dataset_name = "LP_test"
test_images_path = "test"
test_json_annot_path = "test.json"

#we need to save this configuration to later use in test.py
#For object detection
#cfg_save_path = "OD_cfg.pickle"

#For Instance Segmentation
cfg_save_path = "IS_cfg.pickle"
#############################################
#register our dataset 
register_coco_instances(name = train_dataset_name, metadata={}, json_file=train_json_annot_path, image_root=train_images_path)
register_coco_instances(name = test_dataset_name, metadata={}, json_file=test_json_annot_path, image_root=test_images_path)


############################################
#Verify 
#now go utils.py
#because our dataset is registered, but we also need to verify whether or not our annotations are correctly picked up by detectron2
#plot_samples(dataset_name=train_dataset_name, n=2 )

############################################
def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir)

    #save cfg for using test
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)


    #Directories where our output model will be saved
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

if __name__ == '__main__':
    main()

