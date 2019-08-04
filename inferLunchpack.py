#!/usr/bin/env python
"""Semantic segmentation for Coppepan. infer Coppepan image list. Evaluate segmentation accuracy.
"""
from __future__ import print_function
import argparse
import datetime
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy import ndimage
import csv
import cv2

import six
import six.moves.cPickle as pickle
from six.moves import queue

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe
from matplotlib.ticker import *
from chainer import serializers
from chainer import cuda
import chainercv

from lunchpackNet import lunchpackNet

#
# Arguments.
#
parser = argparse.ArgumentParser(
    description='NG Coppepan semantic segmentation')
parser.add_argument('list', help='Path to inference image list file')
parser.add_argument('--out_root', '-R', default='./Segmentation', help='Root directory path of segmentation files')
parser.add_argument('--csv', '-c', default='Accuracy.csv', help='Path to Accuracy text file')
parser.add_argument('--model', '-m', default='model', help='Path to model file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of image files')
parser.add_argument('--RotationFlip', '-rf', default=False, type=bool, help='Flag of image rotation and flip')
args = parser.parse_args()

#
# Preprocessing.
#
model = lunchpackNet()  # Initialize a neural network model.

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

serializers.load_npz(args.model, model)  # Load trained parameters of the model.
csv_path = open(args.csv, 'w')  # Open a output csv file.
csv_writer = csv.writer(csv_path, lineterminator='\n')  # CSV writer.
try:
    os.mkdir(args.out_root)
except FileExistsError:
    pass

"""
## Visualize Filter ============================
# print(model.conv1.W.shape)
# print(model.conv1.W.data[0,0])

n1, n2, h, w = model.conv1.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv1.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# plt.savefig(args.model+'_conv1.png',dsp=150)
# cv2.waitKey(0)

n1, n2, h, w = model.conv2.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(8, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv2.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# cv2.waitKey(0)

n1, n2, h, w = model.conv3.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(n1):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv3.W.data[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

cv2.waitKey(0)
# ============================================
"""

# model.to_cpu()                     # Use CPU for chainer.
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()  # Set a GPU device number.
    model.to_gpu()  # Transfer the model to GPU

chainer.config.cudnn_deterministic = True  # Set deterministic mode for cuDNN.
chainer.config.train = False  # Set inference mode for Chainer.


# Load path pairs of training image data
def load_image_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        assert len(pair) == 2, pair
        tuples.append((os.path.join(root, pair[0]), (os.path.join(root, pair[1]))))
    return tuples


# Prepare the training dataset list
img_list = load_image_list(args.list, args.root)

def read_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Read an input image
    assert image.ndim == 3, image.shape
    assert image.dtype == np.uint8, image.dtype
    return image


def read_label(label_path):
    # label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)    # Read a label image
    # assert label.ndim == 3, label.shape
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)    # Read a label image as gray scale
    assert label.ndim == 2, label.shape
    assert label.dtype == np.uint8, label.dtype
    return label

# Preprocessing for input image and label
def get_pair(img_path, lab_path, augmentation=True):
    image = read_image(img_path)
    label = read_label(lab_path)

    image = image.astype(np.float32) / 255  # Normalize pixel values to 0.0 - 1.0
    label *= 255
    label = label // 128
    label = label.astype(np.int32)  # Convert pixel values to label numbers

    # Data augmentation
    if augmentation:
        if random.random() > 0.5:   # Random flip
            tmp1 = cv2.flip(image, 0)  # Vertical flip
            tmp2 = cv2.flip(label, 0)  # Vertical flip
            # tmp1 = xp.flipud(image)
            # tmp2 = xp.flipud(label)
            image = tmp1
            label = tmp2

        if random.random() > 0.5:   # Random flip
            tmp1 = cv2.flip(image, 1)  # Horizontal flip
            tmp2 = cv2.flip(label, 1)  # Horizontal flip
            # tmp1 = xp.fliplr(image)
            # tmp2 = xp.fliplr(label)
            image = tmp1
            label = tmp2

        angle = random.uniform(0, 360)  # Random rotation angle
        tmp1 = ndimage.interpolation.rotate(image, angle, reshape=False, order=1,
                                            mode='constant', cval=0.0)  # Rotation by bilinear
        tmp2 = ndimage.interpolation.rotate(label, angle, reshape=False, order=0,
                                            mode='constant', cval=0)  # Rotation by nearest neighbor
        image = tmp1
        label = tmp2

    label = cv2.resize(label, dsize=None, fx=1/8, fy=1/8, interpolation=cv2.INTER_NEAREST)
    image = image.transpose(2, 0, 1)

    return image, label


# Convert label to RGB
def label2RGB(label):
    rgb = np.zeros((label.shape[0], label.shape[1], 3), np.uint8)

    # rgb[:, :, 0] = 255 * (label == 0)
    rgb[:, :, 0] = 0
    rgb[:, :, 1] = 255 * (label == 1)
    rgb[:, :, 2] = 255 * (label == 2)

    return rgb


# Convert score to label
def PDF2label(pdf):
    label = np.argmax(pdf, axis=2).astype(np.int32)  # Find a class of maximum score

    return label


#
# Main loop.
#
if __name__ == '__main__':
    csv_writer.writerow(['Number', 'BG IoU', 'NG IoU', 'Mean IoU', 'Pixel accuracy', 'BG accuracy', 'NG accuracy', 'Mean class accuracy', 'Name'])  # Write titles to csv file.

    for idx in range(len(img_list)):  # Index loop.
        print(idx + 1, '/', len(img_list))
        img_path, lab_path = img_list[idx]
        image, label = get_pair(img_path, lab_path, args.RotationFlip)  # Read images.
        image = image[np.newaxis, :]    # Convert dimension to chainer form.
        label = label[np.newaxis, :]    # Convert dimension to chainer form.

        t0 = time.perf_counter() * 1000  # Start timer
        x = chainer.Variable(xp.asarray(image))  # Transfer an image to GPU. Set an image to chainer input.
        hmap = chainer.cuda.to_cpu(model.predict(x).data[0, :, :])  # Calculate foward propagation. Transfer a score to CPU.
        t1 = time.perf_counter() * 1000  # Stop timer
        print('%f[msec]' % (t1 - t0))

        hmap = (hmap * 255).astype(np.uint8).transpose(1, 2, 0)  # Convert a score to RGB form
        r = hmap[:, :, 1]
        b = np.zeros((model.SRCY, model.SRCX), np.uint8)
        g = np.zeros((model.SRCY, model.SRCX), np.uint8)
        pdf = cv2.merge((b, g, r))  # Change NG color to red.
        seg_label = PDF2label(hmap)  # Convert a score to a label.
        seg = label2RGB(seg_label)  # Convert a label to RGB form.
        b, g, r = cv2.split(seg)
        seg = cv2.merge((b, r, g))  # Change NG color to red.

        mini_label = cv2.resize(seg_label, dsize=None, fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_NEAREST)
        result = chainercv.evaluations.eval_semantic_segmentation(mini_label[np.newaxis, :], label[:])   # Evaluate accuracy.
        print(result)

        if len(result['iou']) == 2:
            csv_writer.writerow([idx + 1, result['iou'][0], result['iou'][1], result['miou'], result['pixel_accuracy'], result['class_accuracy'][0], result['class_accuracy'][1], result['mean_class_accuracy'], img_path])  # Write accuracy to csv file.
        else:
            csv_writer.writerow([idx + 1, '', '', result['miou'], result['pixel_accuracy'], '', '', result['mean_class_accuracy'], img_path])  # Write accuracy to csv file.

        dir, file = os.path.split(lab_path)
        # print(dir, file)
        head, tail = os.path.split(dir)
        # print(head, tail)
        # head, tail = os.path.split(head)
        # print(head, tail)
        out_path = args.out_root + '\\' + tail
        # print(out_path)
        # cv2.waitKey()
        try:
            os.mkdir(args.out_root)
        except FileExistsError:
            pass
        try:
            os.mkdir(out_path)
        except FileExistsError:
            pass
        name, ext = os.path.splitext(file)
        cv2.imwrite(out_path + '\\' + name + '_seg.png', seg)  # Output a segmentation image.

        # layer1 = pdf // 4  # Score image
        layer1 = seg // 4   # Segmentation image
        layer2 = read_image(img_path) // 4 * 3    # Read the original input image

        cv2.destroyAllWindows()  # Close all display windows.
        cv2.imshow(img_path, layer1 + layer2)  # Display a superimposed image.

        # cv2.waitKey(0)
        cv2.waitKey(1000)
        # cv2.waitKey(1)

        cv2.imwrite(out_path + '\\' + name + '_reg.png', layer1 + layer2)  # Output the superimposed image.

        print('')

cv2.destroyAllWindows()  # Close all display windows.
csv_path.close()  # Close the output csv file.
