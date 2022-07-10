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
#from zmq import device


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
    #print("All informations:",outputs["instances"])
    o = outputs["instances"]

    #Sonradan yapılan Sql icin
    #idxofClass = [i for i, x in enumerate(list(outputs['instances'].pred_classes)) if x == 0]
    #classes = o.pred_classes[idxofClass]
    #scores = o.scores[idxofClass]
    #boxes = o.pred_boxes[idxofClass]
    #masks = o.pred_masks[idxofClass]

    #print("classes lenght", len(classes))
  


    #print("##########Boxes##########")
    #print("Boxes", boxes)
    #print("Object 1:",outputs["instances"].pred_boxes[0][0])
    #print("Object 1, x:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][0])
    #print("Object 1, y:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][1])
    #print("Object 1, w:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][2])
    #print("Object 1, h:",outputs["instances"].pred_boxes[0][0].tensor.cpu().numpy()[0][3])

    #print("##########CLASS##########")
    #classss = outputs["instances"].pred_classes.to("cpu").numpy()
    #print ("Classs: ", classss)
    #print("Class only one", classss[0])
    #print("classes lenght", len(classes))
    
    #print("##########Score##########")
    #score = outputs["instances"].scores.to("cpu").numpy()
    #print("Score: : ",score)
    #print("Score only one :", score[0])
    Output_from_instances(o)
    
    plt.figure(figsize=(14,10))
    plt.imshow(v.get_image())
    plt.show()
    

#def on video():

#import mysql.connector
#mydb = mysql.connector.connect(
#    host="localhost",
#    database='detectron',
#    user="root",
#    passwd="admin",
#    )
#mycursor = mydb.cursor()
import pandas as pd 
sqlFormula = "INSERT INTO table_detectron (Class,Coordinates,Confiduence) VALUES(%s, %s,%s)"
#mycursor.execute("CREATE TABLE table_detectron (Class VARCHAR(50), Coordinates VARCHAR(50),Confiduence VARCHAR(50))")


def Output_from_instances(o):
    data = []
    classes = o.pred_classes
    classType = o.pred_classes.to("cpu").numpy()
    scores = o.scores.to("cpu").numpy()
    print("###################################")
    print("Koordinatlar")
    for i in range(len(classes)):
        if classType[i] == 0:
            print("Class {} is cat".format(i))

        class_type = "cat"
        obj = o.pred_boxes[i][0].tensor.cpu().numpy()[0]
        print("Object {} is :".format(i),obj)
        print("Scores {}".format(i),scores[i])
        #Pandas DataFrame
        data.append((class_type, #Type
                     scores[i], #Connfi
                     obj, #coordinates
                     )) # time
        print("###################################")
    data = pd.DataFrame(data, columns = ['Type','Confiduence','coordinates'])
    print(data)        
       



