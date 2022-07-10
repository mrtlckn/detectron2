#json dosyaındaki bbox(bounding box information) object detection için
# segmentation yeri ise object segmentation için

# bu nedenle we are able to train both the custom 
#object detection model and the custom instance segmentation model
#because in our annotations(in json) we have both
#of the information


from detectron2.data import DatasetCatalog, MetadataCatalog

#visualize all the annotations and all the predictions
from detectron2.utils.visualizer import Visualizer

#we need config method to load the configuration for object detection model
from detectron2.config import get_cfg

#we can load pretrained model check points for object detection, we need model zoo
from detectron2 import model_zoo

from detectron2.utils.visualizer import ColorMode
#libraries
import random
import cv2
import matplotlib.pyplot as plt
from zmq import device

#function about our annotatins are correctly registered with detectron2
#What function do;
#that will take dataset name 
# and then it will load the datasetfrom detectron2
#and then randomly we'll plot some images
#so we can verify that 
def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:,:,::-1],metadata=dataset_custom_metadata,scale=0.5) #Detectron2 expects the images in RGB
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes,device, output_dir):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir

    return cfg
from detectron2.structures import Instances
from detectron2.data.datasets import register_coco_instances
obj = Instances(image_size=(480, 640))
def on_image(image_path,predictor):

    for d in ["train", "test"]:
        register_coco_instances(f"LP_{d}", {}, f"{d}.json", f"{d}")
    dataset_dicts = DatasetCatalog.get("LP_train")
    dataset_custom_metadata = MetadataCatalog.get("LP_train")
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    v = Visualizer(im[:,:,::-1], metadata=dataset_custom_metadata,scale = 0.5,instance_mode=ColorMode.SEGMENTATION)#dataset_custom_metadata
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #print("Outputs:", outputs) #outputs["instances"] ile aynı 
    print("All informations:",outputs["instances"])
    print("Object 1:",outputs["instances"].pred_boxes[0][0])
    print("Object 1, x:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][0])
    print("Object 1, y:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][1])
    print("Object 1, w:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][2])
    print("Object 1, h:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][3])

    #Sonradan yapılan Sql icin
    idxofClass = [i for i, x in enumerate(list(outputs['instances'].pred_classes)) if x == 0]
    o = outputs["instances"]
    classes = o.pred_classes[idxofClass]
    scores = o.scores[idxofClass]
    boxes = o.pred_boxes[idxofClass]
    masks = o.pred_masks[idxofClass]
    print("idxofClass:", idxofClass)

    print("classes : ", classes)
    print("Classes but only one object: ", o.pred_classes)
    print("classes lenght", len(classes))
    print("Scores : ", scores[0])
    print("boxes", boxes)
    #Classes
    print("##########CLASS##########")
    classss = outputs["instances"].pred_classes.to("cpu").numpy()
    print ("Classs: ", classss)
    #Score
    print("##########Score##########")
    score = outputs["instances"].scores.to("cpu").numpy()
    print("Score: : ",score)
    print("Score only one :", score[0])
    #print("Calisti mi : ", classes, "Scores: ",scores)

    #obj icine atmak
    #obj.set('pred_classes', classes)
    #obj.set('scores', scores)
    #obj.set('pred_boxes', boxes)
    #obj.set('pred_masks', masks)
    
    
   

    #pred_classes
    #print("pred_classes :",outputs["instances"].pred_classes[0][0].tensor.cpu().numpy()[0][0])
    #print("Object 1:",outputs["instances"].num_instances[0])
    
    plt.figure(figsize=(14,10))
    plt.imshow(v.get_image())
    plt.show()
    

#def on video():




