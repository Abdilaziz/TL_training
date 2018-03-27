# This script extracts traffic lights from stills and saves them to a directory

# Heavily inspired by
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import logging
import cv2
import pathlib

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("/opt/tensorflow_models/research/")
sys.path.append("/opt/tensorflow_models/research/slim")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

if tf.__version__ < '1.4.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.4.* or later!')

# What model to download.
MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/opt/tensorflow_models/research/object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# What images to load
PATH_TO_TEST_IMAGES_DIR = '../imgs/camera/'
# Where to store output
DESTINATION_DIR = '../imgs/validation'


def download_model():
    logging.debug('download_model')
    logging.info('Downloading model: ' + MODEL_NAME)
    if not os.path.isfile(PATH_TO_CKPT):
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())


def load_model():
    logging.debug('load_model')
    logging.info('Load TF model to memory')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    logging.info('Load labels')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph


def run_inference_for_single_image(image, sess):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image
        # coordinates and fit the image size.
        real_num_detection = tf.cast(
            tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(
            detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(
            detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def extract_traffic_light(src_images, dst_dir, tf_session):
    logging.debug('extract_traffic_light')
    logging.info('Process images')
    for image_path in src_images:
        logging.info('Working on image ' + image_path)
        src_dir, src_name = os.path.split(image_path)
        image_np = cv2.imread(image_path)
        if image_np is not None:
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np[..., ::-1],
                                                         tf_session)
            detected_tl_boxes = output_dict['detection_boxes'][
                output_dict['detection_classes'] == 10]

            for i, b in enumerate(detected_tl_boxes):
                x = (b[[1, 3]] * image_np.shape[1]).astype(int)
                y = (b[[0, 2]] * image_np.shape[0]).astype(int)
                fn = src_name + '_' + str(i) + '.jpg'
                cv2.imwrite(
                    os.path.join(dst_dir, fn), image_np[y[0]:y[1], x[0]:x[1]])


def main():
    download_model()
    tf_graph = load_model()

    validation_images = [PATH_TO_TEST_IMAGES_DIR + f for f in
                         os.listdir(PATH_TO_TEST_IMAGES_DIR)]
    if not os.path.exists(DESTINATION_DIR):
        pathlib.Path(DESTINATION_DIR).mkdir(parents=True)

    with tf_graph.as_default():
        with tf.Session() as tf_sess:
            extract_traffic_light(validation_images, DESTINATION_DIR, tf_sess)


if __name__ == '__main__':
    main()
