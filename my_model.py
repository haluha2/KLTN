import os
from keras.models import Model
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam, SGD, RMSprop
from ultility.layer import *

def load_Faster_RCNN_model(C):

    num_features = 512

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = rpn_layer(shared_layers, num_anchors)

    classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    print('Loading weights from {}'.format(C.model_path))
    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')
    return model_rpn, model_classifier
