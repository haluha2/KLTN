import os, sys, time, cv2
from annoy import AnnoyIndex
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam, SGD, RMSprop
import ultility.config
import ultility.layer
import ultility.ultility
import ultility.prepare_data

class Search_Image():
    def __init__(self, C, **kwargs):

        self.base_path = './'
        self.test_path = 'data/test_annotation.txt' # Test data (annotation file)
        self.test_base_path = 'BITVehicle_Dataset' # Directory to save the test images

        config_output_filename = 'data/model_vgg_config.pickle'
        self.C = C

        # turn off any data augmentation at test time
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False

        self.num_features = 512

        self.input_shape_img = (None, None, 3)
        self.input_shape_features = (None, None, num_features)

        self.img_input = Input(shape=input_shape_img)
        self.roi_input = Input(shape=(C.num_rois, 4))
        self.feature_map_input = Input(shape=input_shape_features)

        # define the base network (VGG here, can be Resnet50, Inception, etc)
        self.shared_layers = nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        self.num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        self.rpn_layers = rpn_layer(shared_layers, num_anchors)

        self.classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier = Model([feature_map_input, roi_input], classifier)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)
        self.model_extract_feature_only = Model(model_classifier_only.input, model_classifier_only.get_layer(model_classifier_only.layers[-3].name).output)
        load_weight(self.C.model_path)
        

        super(Search_Image, self).__init__(**kwargs)

    def load_weight(self, weight_file):
        print('Loading weights from {}'.format(weight_file))
        self.model_rpn.load_weights(weight_file, by_name=True)
        self.model_classifier.load_weights(weight_file, by_name=True)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        self.model_classifier.compile(optimizer='sgd', loss='mse')

    def crop_feature(img, bbox_threshold = 0.7, verbose=False):
        ''' The function predict the label and apply the bbox of the vehicles.
        Args:
            imgs_path: List/array of the test img.
            test_base_path: Folder contain imgs. (Default=None)
            vehicles: List/array of the labels and bboxes each img. (Default=None)
            bbox_threshold: If the box classification value is less than this,
            we ignore this box. (Default = 0.7)
        '''
        
        # Check is image file.
        if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
            return "Wrong format"

        st = time.time() # Start count time to predict
##        filepath = img_name
##        if test_base_path is not None:
##            filepath = os.path.join(test_base_path, img_name)
##        # Read the img
##        img = cv2.imread(filepath) # img.shape=(1200, 1600, 3)

        # Resize img to input model size.
        # Return img resized and the ratio resized.
        # e.g: img(1200, 1600) => img_resized(600,800), ratio = 0.5
        X, ratio = format_img(img, self.C) # X.shape=(1, 3, 600, 800), ratio=0.5
        # Format the img
        X = np.transpose(X, (0, 2, 3, 1)) # X.shape=(1, 600, 800, 3)

        # get output layer Y1, Y2 from the RPN and the feature maps F
        # Y1: y_rpn_cls
        # Y2: y_rpn_regr
        # Y1.shape = (1, 37, 50, 9)
        # Y2.shape = (1, 37, 50, 36)
        # F.shape = (1, 37, 50, 512)
        [Y1, Y2, F] = model_rpn.predict(X)

        # Get bboxes by applying NMS 
        # R.shape = (300, 4)
        # 4 = (x1,y1,x2,y2)
        # It mean R contain 300 couple of points (x1,y1,x2,y2)
        R = rpn_to_roi(Y1, Y2, C, K.common.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        
        # Switch key value for class mapping We need type: {0:Bus,1:Sedan,...}
        class_mapping = self.C.class_mapping
        class_mapping = {v: k for k, v in class_mapping.items()}
        # apply the spatial pyramid pooling to the proposed regions
        # bboxes of objects in the img.
        bboxes = {} # e.g: {'Sedan': [0.98768216, 0.7442094, ..., 0.9753626, 0.9117886, 0.9728151]}
        # probability of the object class in bboxes
        probs = {} # e.g: {'Sedan': [[320, 144, 608, 432], [336, 144, 640, 432], ..., [336, 144, 640, 448], [320, 144, 608, 416], [336, 144, 640, 432]]}
        # len(bboxes) = len(probs)
        features = {}
        

        # Predict bboxes and classname
        for jk in range(R.shape[0]//self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            if jk == R.shape[0]//self.C.num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
                P_features = model_extract_feature_only.predict([F, ROIs])
                # print(P_cls.shape)
                # print(P_features.shape)

                # Calculate bboxes coordinates on resized image
                for ii in range(P_cls.shape[1]):
                    # Ignore 'bg' class
                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                    # Add class if not exist in bboxes
                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []
                        features[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= self.C.classifier_regr_std[0]
                        ty /= self.C.classifier_regr_std[1]
                        tw /= self.C.classifier_regr_std[2]
                        th /= self.C.classifier_regr_std[3]
                        x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))
                    features[cls_name].append(P_features[0][ii])

        # all_dets contain the results.
        # [('Sedan', 99.34011101722717)]
        all_dets = []


        for key in bboxes:
            bbox = np.array(bboxes[key])
            prob = np.array(probs[key])
            for bb in range(bbox.shape[0]):
                new_probs = prob[bb]
                feat = features[key][bb]
                (x1, y1, x2, y2) = bbox[bb]

                # Calculate real coordinates on original image
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                textLabel = '{}: {}'.format(key,int(100*new_probs))
                all_dets.append((key,100*new_probs,[real_x1, real_y1, real_x2, real_y2], feat))
                # obj_color = (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2]))
                # img = apply_bbox(img, textLabel, (real_x1, real_y1, real_x2, real_y2), color=obj_color)


        if verbose == True:
            print('Elapsed time = {}'.format(time.time() - st))
        return all_dets
