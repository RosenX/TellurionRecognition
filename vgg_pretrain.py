from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import h5py
import os, pickle, sys

def VGG_16_pretrain(weights_path=None, img_width=128, img_height=128, channel = 3):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channel, img_width, img_height)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    weights_path = 'data/vgg16_weights.h5'

    f = h5py.File(weights_path)
    for k in range(len(model.layers)):
        if isinstance(model.layers[k],Convolution2D):
            attr = model.layers[k].name
            g = f[attr]
            weights = [g[attr+'_W'], g[attr+'_b']]
            model.layers[k].set_weights(weights)
    f.close()

    print('Model loaded.')

    return model

if __name__ == "__main__":

    # config dir
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    class_num = 10
    vgg_img_width, vgg_img_height, vgg_channel = (224, 224, 3)

    # define model
    VGG_model = VGG_16_pretrain('data/vgg16_weights.h5', vgg_img_width, vgg_img_height, vgg_channel)
    sgd = SGD(lr = 1e-5, decay = 0.0, momentum = 0.9, nesterov=True)
    VGG_model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

    # pretrain
    have_load = 0; img_count = 0
    total = len(os.listdir(source_dir))
    for f in os.listdir(source_dir):
        pic_path = source_dir + '/%s' % f
        # c = int(f.split('.')[0]) / 100
        c = int(f.split('_')[0])
        im = cv2.resize(cv2.imread(pic_path), (vgg_img_width, vgg_img_height)).astype(np.float32) 
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis = 0)

        X = VGG_model.predict(im)

        with open(dest_dir + '/%d_img_%d.pickle' % (c, img_count), 'wb') as f:
            pickle.dump(X, f)

        have_load += 1; img_count += 1
        print '%6d/%6d' % (img_count, total)

          