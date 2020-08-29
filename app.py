from flask import Flask, jsonify, session, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
import os, shutil
import cv2
import numpy as np
import pandas as pd
import datetime, time
import pickle
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.models import Model
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, save_img
from annoy import AnnoyIndex
from my_model import *
from ultility.config import *
from ultility.prepare_data import *
from ultility.ultility import *

APP_NAME = "flask_app"
UPLOAD_FOLDER = './static/img/upload_images'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Change to GPU
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_rpn = None
model_classifier = None
model_extract_feature_only = None
train_imgs, classes_count, class_mapping = load_saved_data("Saved_BIT_Vehicle")
df_all = pd.read_csv("./data/manage_image.csv")
featureDB = pd.read_csv("./data/Df_Search.csv")
f = 4096
vehicle_query = AnnoyIndex(f,'euclidean')
vehicle_query.load("./data/Db_feature(model).ann")

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "flask_app"
        return super().find_class(module, name)
C = None
##config_output_filename = os.path.join("./data", 'model_vgg_config.pickle')
##with open(config_output_filename, 'rb') as f_in:
##    unpicker = MyCustomUnpickler(f_in)
##    C = unpicker.load()
config_output_filename = os.path.join("./data", 'model_vgg_config.pickle')
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)
C.model_path = "./model/model_frcnn_vgg.hdf5"
sess = tf.Session()
set_session(sess)
model_rpn, model_classifier = load_Faster_RCNN_model(C)
out_cls = model_classifier.get_layer(model_classifier.layers[-2].name).output
out_fea = model_classifier.get_layer(model_classifier.layers[-3].name).output
model2 = Model(model_classifier.input, [out_cls,out_fea])

global graph
graph = tf.get_default_graph()
def convertRGBtoBGR(img_arr):
    ''' OpenCV reads images in BGR format whereas in keras,
    it is represented in RGB.
    This function convert img from BGR to RGB and vice versa
    '''
    return img_arr[...,::-1]

def crop_feature(img, bbox_threshold = 0.7, verbose=True):
  ''' The function predict the label and apply the bbox of the vehicles.

  Args:
    img: Input img.
    bbox_threshold: If the box classification value is less than this,
    we ignore this box. (Default = 0.7)
    verbose: Show time extract feature.
  Return:
    list(prob, feat) 
    prob: class probabily of vector
    feat: vector 4096D
  '''
  class_mapping = {v: k for k, v in C.class_mapping.items()}
  st = time.time() # Start count time to predict
  #img = cv2.imread(filepath) # img.shape=(1200, 1600, 3)

  # Resize img to input model size.
  # Return img resized and the ratio resized.
  # e.g: img(1200, 1600) => img_resized(600,800), ratio = 0.5
  X, ratio = format_img(img, C) # X.shape=(1, 3, 600, 800), ratio=0.5
  # Format the img
  X = np.transpose(X, (0, 2, 3, 1)) # X.shape=(1, 600, 800, 3)

  # get output layer Y1, Y2 from the RPN and the feature maps F
  # Y1: y_rpn_cls
  # Y2: y_rpn_regr
  # Y1.shape = (1, 37, 50, 9)
  # Y2.shape = (1, 37, 50, 36)
  # F.shape = (1, 37, 50, 512)
  with graph.as_default():
      set_session(sess)
      [Y1, Y2, F] = model_rpn.predict(X)

  # Get bboxes by applying NMS 
  # R.shape = (300, 4)
  # 4 = (x1,y1,x2,y2)
  # It mean R contain 300 couple of points (x1,y1,x2,y2)
  R = rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

  # convert from (x1,y1,x2,y2) to (x,y,w,h)
  R[:, 2] -= R[:, 0]
  R[:, 3] -= R[:, 1]

  # apply the spatial pyramid pooling to the proposed regions
  # probability of the object class in bboxes
  probs = {} # e.g: {'Sedan': [[320, 144, 608, 432], [336, 144, 640, 432], ..., [336, 144, 640, 448], [320, 144, 608, 416], [336, 144, 640, 432]]}
  # len(bboxes) = len(probs)
  features = {}

  # Predict bboxes and classname
  for jk in range(R.shape[0]//C.num_rois + 1):
      ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
      if ROIs.shape[1] == 0:
          break
      if jk == R.shape[0]//C.num_rois:
        #pad R
        curr_shape = ROIs.shape
        target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
        ROIs_padded[:, :curr_shape[1], :] = ROIs
        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
        ROIs = ROIs_padded
      
      with graph.as_default():
          set_session(sess)
          [P_cls, P_features] = model2.predict([F, ROIs])
      # print(P_cls.shape)
      # print(P_features.shape)

      # Calculate bboxes coordinates on resized image
      for ii in range(P_cls.shape[1]):
        # Ignore 'bg' class
        if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
            continue

        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
        # Add class if not exist in bboxes
        if cls_name not in probs:
            probs[cls_name] = []
            features[cls_name] = []

        probs[cls_name].append(np.max(P_cls[0, ii, :]))
        features[cls_name].append(P_features[0][ii])

  all_dets = []


  for key in probs:
      prob = np.array(probs[key])
      for bb in range(prob.shape[0]):

        new_probs = prob[bb]
        feat = features[key][bb]

        all_dets.append((100*new_probs, feat))
      

  if verbose == True:
    print('Elapsed time = {}'.format(time.time() - st))
  return all_dets

##def paging(_page):
##    return jsonify(result_imgs[9*(_page-1):9*_page -1])
def search_vehicles(img, disitance_function, threshold=0.5):
    result_imgs = []
    feats = crop_feature(img, verbose=False)
    if len(feats)<1:
        return result_imgs
    max_feat = feats[np.argmax([feats[i][0] for i in range(len(feats))])] # Get feature with the highest prob
    feature = max_feat[1].flatten()
    vehicle_query_result = vehicle_query.get_nns_by_vector(feature, n=500)
    selected_result = [i for i in vehicle_query_result if disitance_function(feature,vehicle_query.get_item_vector(i)) < threshold]
    selected_result = sorted(selected_result)
    return_imgs = featureDB.iloc[selected_result]['name'].tolist()
    df_return = df_all[df_all['name'].isin(return_imgs)]
    return [{"img":row['name'],"id":row['img_id'],"width":row['width'],"height":row["height"]} for index, row in df_return.iterrows()]

app = Flask("flask_app", static_folder='static')
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result_imgs = []
    current_img = None
    return app.send_static_file('html/index.html')

@app.route('/path')
def path():
    return os.getcwd()

@app.route('/prediction')
def prediction():
    page = (int)(request.args.get('page'))
    if page is None:
        page=1
    if page > 1:
        pass
    else:
        if current_img is None:
            return ""
        img = cv2.imread(current_img)
        feats = crop_feature(img, verbose=False)
        max_feat = feats[np.argmax([feats[i][0] for i in range(len(feats))])] # Get feature with the highest prob
        vehicle_query_result = vehicle_query.get_nns_by_vector(max_feat[1], n=500)
        result_imgs = featureDB.iloc[vehicle_query_result]['name']
    return paging(page)
@app.route("/upload", methods=['POST'])
def upload():
    fileUpload = request.files['files']

    SAVE_PATH = "./static/img/upload_images"

##    #read image file string data
    filestr = fileUpload.read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
##    img = cv2.imread(fileUpload)
##    img = convertRGBtoBGR(img)
##    img = cv2.resize(img,(resized_shape[1], resized_shape[0]))

##    predict_img = model.predict(img/255)

    name = (int)(datetime.datetime.utcnow().timestamp())
    
    # Create folder if not existed
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    result = f"{SAVE_PATH}/{name}.png"
    
    #save_img(result,predict_img)
    cv2.imwrite(result,img)
    return jsonify(search_vehicles(img,euclidean))

if __name__ == '__main__':
    app.run(debug=True)
