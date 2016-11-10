#coding = utf8

import os
import random, shutil

def splitCorel():
    train_set = 'data/corel_train'
    test_set = 'data/corel_test'
    data_set = 'data/corel1000'
    train_proportion = 0.7
    class_num = 10

    for c in range(1, class_num + 1):
        folder = data_set + '/%d/' % c
        pic_list = os.listdir(folder)
        random.shuffle(pic_list)
        split_index = int(train_proportion * len(pic_list))
        train_list = pic_list[0 : split_index]
        test_list = pic_list[split_index : ]
        for pic in train_list:
            old_path = folder + '/%s' % pic
            new_path = train_set + '/%s' % pic
            shutil.copyfile(old_path, new_path)
        for pic in test_list:
            old_path = folder + '/%s' % pic
            new_path = test_set + '/%s' % pic
            shutil.copyfile(old_path, new_path)

def splitNation():
    # data_set = 'data/Nation'
    # for index, folder_name in enumerate(os.listdir(data_set)):
    #     folder = data_set + '/' + folder_name
    #     for pic in os.listdir(folder):
    #         os.rename(folder + '/' + pic, folder + '/%d_%s' % (index, pic))
    split('data/Nation', 'data/nation_train', 'data/nation_test', prop = 0.7)

def split(data_set, train_set, test_set, prop = 0.7, size = 1000):
    for folder_name in os.listdir(data_set):
        folder = data_set + '/' + folder_name
        pic_list = os.listdir(folder)
        random.shuffle(pic_list)
        if size != -1: pic_list = pic_list[0:size]
        split_index = int(prop * len(pic_list))
        train_list = pic_list[0 : split_index]
        test_list = pic_list[split_index : ]
        for pic in train_list:
            old_path = folder + '/%s' % pic
            new_path = train_set + '/%s' % pic
            shutil.copyfile(old_path, new_path)
        for pic in test_list:
            old_path = folder + '/%s' % pic
            new_path = test_set + '/%s' % pic
            shutil.copyfile(old_path, new_path)

def splitCaltech():
    # data_set = 'data/caltech-101'
    # for index, folder_name in enumerate(os.listdir(data_set)):
    #     folder = data_set + '/' + folder_name
    #     for pic in os.listdir(folder):
    #         os.rename(folder + '/' + pic, folder + '/%d_%s' % (index, pic))
    split('data/caltech-101', 'data/caltech_train', 'data/caltech_test', prop = 0.7)

def splitBigData():
    # data_set = 'data/BigData'
    # for index, folder_name in enumerate(os.listdir(data_set)):
    #     folder = data_set + '/' + folder_name
    #     for pic in os.listdir(folder):
    #         os.rename(folder + '/' + pic, folder + '/%d_%s' % (index, pic))
    split('data/BigData', 'data/bigdata_train', 'data/bigdata_test', prop = 0.8, size = -1)

def splitData(origin, train, test, prop, size):
    if not os.path.exists(train): os.mkdir(train)
    if not os.path.exists(test): os.mkdir(test)
    # for index, folder_name in enumerate(os.listdir(origin)):
    #     folder = origin + '/' + folder_name
    #     for pic in os.listdir(folder):
    #         os.rename(folder + '/' + pic, folder + '/%d_%s' % (index, pic))
    split(origin, train, test, prop, size)

def sample(origin_dir, sampled_dir, rate):
    for folder_name in os.listdir(origin_dir):
        folder = origin_dir + '/' + folder_name
        new_folder = sampled_dir + '/' + folder_name
        if not os.path.exists(new_folder): os.mkdir(new_folder)
        pic_list = os.listdir(folder)
        random.shuffle(pic_list)
        select_list = os.listdir(new_folder)
        for pic in pic_list:
            if len(select_list) == rate: break
            if pic not in select_list: select_list.append(pic)

        for pic in select_list:
            old_path = folder + '/%s' % pic
            new_path = new_folder + '/%s' % pic
            shutil.copyfile(old_path, new_path)

def sample(origin_dir, exist_sampled_dir, new_sampled_dir, rate):
    if not os.path.exists(new_sampled_dir): os.mkdir(new_sampled_dir)
    for folder_name in os.listdir(origin_dir):
        folder = origin_dir + '/' + folder_name
        new_folder = exist_sampled_dir + '/' + folder_name
        add_folder = new_sampled_dir + '/' + folder_name
        if not os.path.exists(new_folder): os.mkdir(new_folder)
        if not os.path.exists(add_folder): os.mkdir(add_folder)
        pic_list = os.listdir(folder)
        random.shuffle(pic_list)
        exist_list = os.listdir(new_folder)
        select_list = []
        for pic in pic_list:
            if len(select_list) == rate - len(exist_list): break
            if pic not in exist_list: select_list.append(pic)

        for pic in select_list:
            old_path = folder + '/%s' % pic
            new_path = add_folder + '/%s' % pic
            shutil.copyfile(old_path, new_path)

def rename(path):
    for folder_name in os.listdir(path):
        folder = path + '/' + folder_name
        for index, pic in enumerate(os.listdir(folder)):
            os.rename(folder + '/' + pic, folder + '/%s_%d' % (pic, index))

def merge(source, dest):
    for folder_name in os.listdir(source):
        print folder_name
        folder = source + '/' + folder_name
        new_folder = dest + '/' + folder_name
        count = 0
        for pic in os.listdir(folder):
            count += 1
            if count % 50 == 0: print count
            old_path = folder + '/' + pic
            new_path = new_folder + '/' + pic
            shutil.copyfile(old_path, new_path)

if __name__ == '__main__':
    # splitNation()
    # splitCaltech()
    # splitData('data/data4', 'data/data4_train', 'data/data4_test', prop = 0.7, size = -1)
    # rename('data/total_sample_1_vgg')
    # merge('data/Total_Sampled_Vgg', 'data/total_sample_1_vgg')
    # splitData('data/', 'data/total_train', 'data/total_test', prop = 0.8, size = -1)
    # sample('data/total', 'data/total_sample', 'data/total_sample_1', 1000)
    splitData('data/total_sample_vgg', 'preTrainData/total_sampled_train', 'preTrainData/total_sampled_test', 
              prop = 0.8, size = -1)
            
            
            