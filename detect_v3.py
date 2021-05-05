import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils_v2 as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('weights', '/content/gdrive/MyDrive/Yolov4/checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('images', '/content/gdrive/MyDrive/Yolov4/data/images/kite.jpg', 'path to input image')
flags.DEFINE_string('output', '/content/gdrive/MyDrive/Yolov4/detections/', 'path to output folder')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = FLAGS.size
    image_path = FLAGS.images
    count = 1

    # load model
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    #for count, image_path in enumerate(images, 1):
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    
    # get image name by using split method
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    #load model
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    
    # custom allowed classes (uncomment line below to allow detections for only people)
    #allowed_classes = ['person']
       
    image = utils.draw_bbox(original_image, pred_bbox, FLAGS.info, allowed_classes=allowed_classes, read_plate = FLAGS.plate)        
    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output + 'detection9.png', image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
