from detectron2.engine import DefaultPredictor

import os
import pickle

from utils import *

#cfg_save_path = "OD_cfg.pickle"
cfg_save_path = "IS_cfg.pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

#Load weight
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

#threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
train_dataset_name= "LP_train"
predictor = DefaultPredictor(cfg)
image_path = "test/dog-and-owner.jpg"
on_image(image_path, predictor)


