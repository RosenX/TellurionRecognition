from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2
import numpy as np
import h5py
import os

def VGG_16(weights_path=None, img_width=128, img_height=128, channel = 3):
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

    sgd = SGD(lr = 1e-5, decay = 0.0, momentum = 0.9, nesterov=True)
    model.compile(optimizer = sgd, loss='categorical_crossentropy', metrics = ['accuracy'])

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    f = h5py.File(weights_path)
    for k in range(len(model.layers)):
        if isinstance(model.layers[k],Convolution2D):
            attr = model.layers[k].name
            g = f[attr]
            weights = [g[attr+'_W'], g[attr+'_b']]
            model.layers[k].set_weights(weights)
    f.close()

    print('VGG16 Model loaded.')

    return model

def NeLet5(weight_path = None, let_img_width = 32, let_img_height = 32, let_channel = 3, class_num = 10):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(let_channel, let_img_width, let_img_height),
                            activation='relu'))
    model.add(Convolution2D(64, 3, 3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_num, activation='softmax'))

    sgd = SGD(lr = 1e-2, decay = 1e-6, momentum = 0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    if weight_path: model.load_weights(weight_path)
    print 'NeLet model loaded.'
    return model

if __name__ == "__main__":
    
    # data
    test_dir = 'test/713images'
    # test_dir = 'test/gray'
    vgg_weight_path = 'data/vgg16_weights.h5'
    # let_weight_path = 'vgg_pretrain_weight_total_sampled.h5'
    let_weight_path = 'vgg_pretrain_weight_total.h5'
    class_num = 13
    vgg_img_width, vgg_img_height, vgg_channel = (224, 224, 3)
    let_img_width, let_img_height, let_channel = (14, 14, 512)

    # pretrain
    VGG_model = VGG_16(vgg_weight_path, vgg_img_width, vgg_img_height, vgg_channel)
    NeLet_model = NeLet5(let_weight_path, let_img_width, let_img_height, let_channel, class_num = class_num)

    Nation2Class = {'America' : 8, 'Australia' : 1, 'Brazil' : 12, 'Canada' : 11, 'China': 0,
                    'England' : 6, 'France' : 7, 'India' : 10, 'Japan' : 3, 'Mexico' : 9, 'Russia' : 5,
                    'SaudiArabia' : 4, 'SouthAfrica' : 2}
    right_pic = 0; all_pic = 0
    for folder_name in os.listdir(test_dir):
        c = Nation2Class[folder_name]
        folder = test_dir + '/' + folder_name
        eachclass_right = 0; eachclass_all = len(os.listdir(folder))
        for pic in os.listdir(folder):
            all_pic += 1
            pic_path = folder + '/' + pic
            img = cv2.resize(cv2.imread(pic_path), (vgg_img_width, vgg_img_height)).astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis = 0)
            new_img = VGG_model.predict(img)
            predict_c = np.argmax(NeLet_model.predict(new_img), axis = 1)
            if predict_c == c:
                eachclass_right += 1
                right_pic += 1
        if eachclass_all:
            print folder_name, float(eachclass_right) / eachclass_all
    
    print 'Total accuracy', float(right_pic) / all_pic