import os
import glob
import shutil
import json
import random
import cv2

import copy
from lxml.etree import Element, SubElement, tostring, ElementTree

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["person","work_shoe","slipper"]


def get_img_from_txt(images_dir, labels_dir, new_images_dir):
    for label_path in glob.glob(labels_dir + '/*.txt'):
        label_name = label_path.split('\\')[-1]
        shutil.copy(os.path.join(images_dir, label_name[:-3] + 'jpg'),
                    os.path.join(new_images_dir, label_name[:-3] + 'jpg'))


def get_inferdata(txt_path, output_path, valid_path):
    paths = open(txt_path)
    for line in paths:
        line = line.strip().split('/')[-1][: -3]
        shutil.copy(os.path.join(output_path, line + 'png'), os.path.join(valid_path, line + 'png'))


def split_with_ratio(all_list, shuffle=False, ratio=0.8):
    """
    随机按比例拆分数据
    """
    num = len(all_list)
    offset = int(num * ratio)
    if num == 0 or offset < 1:
        return [], all_list
    if shuffle:
        random.shuffle(all_list)  # 列表随机排序
    train = all_list[:offset]
    valid = all_list[offset:]
    return train, valid


def write_to_txt(sublist, images_path, path):
    with open(path, 'w') as fw:
        for i in sublist:
            image_path = images_path + '/' + i
            fw.write(image_path + '\n')


def split_trainset_validset(images_path, train_path, valid_path):
    """
    将原始数据集分成验证集和训练集，训练集和验证集的比例是8：2
    :param images_path: 存放所有图像文件的文件夹名
    :param train_path: 记录所有训练集图像文件的文本文件名
    :param valid_path: 记录所有验证集图像文件的文本文件名
    """
    datalist = os.listdir(images_path)
    train_list, valid_list = split_with_ratio(datalist, shuffle=True, ratio=0.8)
    write_to_txt(train_list, 'data/custom/images', train_path)
    write_to_txt(valid_list, 'data/custom/images', valid_path)

    print(len(train_list))
    print(train_list)

def xml_label2txtlabel(xmls_path, txts_path, images_path):
    """
    将xml文件中包含的label信息转化为txt格式的信息
    :param xmls_path:存放xml文件的文件夹名
    :param txts_path:存放txt文件的文件夹名
    """
    # 如果存放txt文件的文件夹不存在，新建一个。
    if not os.path.exists(txts_path):
        os.makedirs(txts_path)

    # 遍历所有xml文件将对应的标签信息和边界框写入同名的txt文件中
    for xml_path in glob.glob(xmls_path+'/*'):
        xml_path_name = os.path.split(xml_path)[-1]
        txt_path = os.path.join(txts_path, xml_path_name[: -4] + '.txt')
        print(xml_path,txt_path)
        # 获取对应json文件的图像路径
        with open(xml_path, 'r') as f:
            label_xml = ET.parse(f).getroot() # 读取xml文件
            fully_width = int(label_xml.find('size').find('width').text)
            fully_height = int(label_xml.find('size').find('height').text)
            print(fully_height,fully_width)

            with open(txt_path, 'w') as fw:
                # 获取每个目标对应的文本框的位置信息，包括中心点的坐标以及物体的宽度和高度。
                # 坐标和宽度高度都是相对于图像整体的，范围在0-1。
                for obj in label_xml.iter('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        continue

                    xmin = float(obj.find('bndbox').find('xmin').text)
                    xmax = float(obj.find('bndbox').find('xmax').text)
                    ymin = float(obj.find('bndbox').find('ymin').text)
                    ymax = float(obj.find('bndbox').find('ymax').text)

                    x = (xmin + xmax) / (2 * fully_width)
                    y = (ymin + ymax) / (2 * fully_height)
                    width = (xmax - xmin) / fully_width
                    height = (ymax - ymin) / fully_height

                    if cls == 'person':
                        fw.write('person ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')
                    elif cls == "work_shoe":
                        fw.write('work_shoe ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')
                    elif cls == "slipper":
                        fw.write('slipper ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')




                    # x = (instance['points'][0][0]+instance['points'][2][0])/(2*fully_width)
                    # y = (instance['points'][0][1]+instance['points'][2][1])/(2*fully_height)
                    # width = abs((instance['points'][2][0] - instance['points'][0][0])/fully_width)
                    # height = abs((instance['points'][2][1] - instance['points'][0][1])/fully_height)
                    # if instance['label'] == '0':
                    #     fw.write('0 ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')
                    # elif instance['label'] == '1':
                    #     fw.write('1 ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + '\n')

def split_images_and_labels(orig_data_path, images_path, xmls_path, prefix):
    """
    将json文件和图像文件从原始文件中分离出来，分别放在images文件夹和jsons文件夹中
    :param orig_data_path: 原始文件夹名
    :param images_path: 存放图像的文件夹名
    :param xml_path: 存放xml文件的文件夹名
    :param prefix: 作为xml文件名和图像文件名的前缀
    """
    # 如果文件夹不存在，新建文件夹
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(xmls_path):
        os.makedirs(xmls_path)

    # 遍历文件夹中所有json文件，将json文件复制到jsons文件夹中，
    # 同时将和json文件名相同文件名的图像文件复制到images文件夹中,
    # 并重新修改json文件和图像文件的文件名
    n = 0
    for xml_path in glob.glob(orig_data_path+'/*.xml'):
        splited_path = os.path.split(xml_path)
        try:
            shutil.copy(os.path.join(splited_path[0], splited_path[1][: -4]+'.jpg'),
                        os.path.join(images_path, prefix+str(n)+'.jpg'))
            shutil.copy(xml_path,
                        os.path.join(xmls_path, prefix+str(n)+'.xml'))  # 修改splited_path[1]为splited_path[-1] 2020/1/7
        except Exception as e:
            print('xml path: {} error is {}'.format(xml_path, e))
        n += 1

if __name__ == '__main__':
    split_images_and_labels(r'dataset', r'images', r'xmls', '20200426_')
    xml_label2txtlabel(r'xmls', r'txts', 'images')
    split_trainset_validset(r'images', 'train.txt', 'valid.txt')



