"""Train Semantic Segmentation for Coppepan"""
import argparse
import logging.config
import multiprocessing
import os
import random
from scipy import ndimage

import chainer
import cv2
import numpy as np
from chainer import cuda
from chainer.training import extensions
import chainercv

from lunchpackNet import lunchpackNet

SNAPSHOT_TRIGGER = (10, 'epoch')    # snapshot period


# WEIGHT_DECAY_PER_EPOCH = 0.5  # weight decay rate per epoch


# Set the arguments
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training coppepanNet')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--val_batchsize', '-b', type=int, default=200,
                        help='Validation minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='Number of epochs to learn')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--loaderjob', '-j', default=15, type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--class_weight', '-w', default=1, type=float,
                        help='NG class weight')
    parser.add_argument('--root', '-r', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--resume',
                        help='Restart training from a given snapshot file')
    parser.add_argument('--out', '-o', default='result',
                        help='Root path to save files')
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    return parser.parse_args(args)


xp = None


def setup_gpu(gpu):
    global xp

    if gpu < 0:
        xp = np  # Use numpy
    else:
        xp = cuda.cupy  # Use cupy
        cuda.check_cuda_available()
        cuda.get_device_from_id(gpu).use()  # Set a GPU device number.


def load_model(class_weight):
    global xp

    model = lunchpackNet()  # Set a neural network model

    if xp == cuda.cupy:
        model = model.to_gpu()  # Transfer a model to GPU
    model.class_weight = xp.array([1.0, class_weight], dtype=xp.float32)  # Set class weights

    return model


# Load path pairs of training image data
def load_path_pairs(image_list_path, root):
    with open(image_list_path) as fp:
        for line in fp:
            if not line.strip():
                continue
            path_pair = line.strip().split()
            assert len(path_pair) == 2, path_pair

            yield tuple(os.path.join(root, path) for path in path_pair)


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


class Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, path_pairs, for_validation=False):
        self._path_pairs = list(path_pairs)
        self._for_validation = for_validation

    # Read image pair
    def _get_raw_pair(self, i):
        image_path, label_path = self._path_pairs[i]    # Get training image paths
        image = read_image(image_path)
        label = read_label(label_path)
        return image, label

    # Preprocess and convert image pair for data augmentation
    def get_example(self, i):
        image, label = self._get_raw_pair(i)    # Get image pair
        # cv2.imshow('label', label)
        # cv2.waitKey()

        image = image.astype(np.float32) / 255  # Normalize pixel values to 0.0 - 1.0
        # label = 0 if label == 0 else 1
        # label = 1 * (label != 0)
        label *= 255
        label = label//128
        # cv2.imshow('label', label*255)
        # cv2.waitKey()
        label = label.astype(np.int32)   # Convert pixel values to label numbers

        if self._for_validation:
            pass
        else:   # Data augmentation
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
            tmp1 = ndimage.interpolation.rotate(image, angle, reshape=False, order=1, mode='constant',
                                                cval=0.0)  # Rotation by bilinear
            tmp2 = ndimage.interpolation.rotate(label, angle, reshape=False, order=0, mode='constant',
                                                cval=0)  # Rotation by nearest neighbor
            image = tmp1
            label = tmp2

        # cv2.imshow('label', label*2147483647)
        # cv2.waitKey()
        label = cv2.resize(label, dsize=None, fx=1/8, fy=1/8, interpolation=cv2.INTER_NEAREST)
        # cv2.imshow('label', label*2147483647)
        # cv2.waitKey()
        image = image.transpose(2, 0, 1)
        # label = label.transpose(1, 0)
        return image, label

    def __len__(self):
        return len(self._path_pairs)


"""
class ClassWeightUpdater(chainer.training.Extension):
    def __init__(self, model, trigger=(1, 'iteration')):
        self._model = model
        self._trigger = chainer.training.trigger.get_trigger(trigger)
        self._decay_count = 0

    def _update(self, epoch):
        self._model.class_weight[0] = 1
        self._model.class_weight[1] = 1 + 44 * WEIGHT_DECAY_PER_EPOCH ** epoch

    def __call__(self, trainer):
        if self._trigger(trainer):
            with chainer.cuda.get_device_from_array(self._model.class_weight):
                self._update(epoch=trainer.updater.epoch_detail)
"""


def _main():
    args = parse_args()
    setup_gpu(gpu=args.gpu)
    assert xp is not None

    # Model
    model = load_model(class_weight=args.class_weight)    # Load a neural network model
    # optimizer = chainer.optimizers.Adam(0.001)
    optimizer = chainer.optimizers.Adam()   # set an optimization function as Adam method
    optimizer.setup(model)

    # Dataset
    # Multiprocess data loading iterator with shuffle order for training
    train_iter = chainer.iterators.MultiprocessIterator(
        Dataset(load_path_pairs(args.train, args.root)),
        batch_size=args.batchsize, n_processes=args.loaderjob)

    # Multiprocess data loading iterator with fixed order for evaluation
    test_iter = chainer.iterators.MultiprocessIterator(
        Dataset(load_path_pairs(args.val, args.root), for_validation=True),
        batch_size=args.val_batchsize, n_processes=args.loaderjob, repeat=False, shuffle=False)

    # Updater, Trainer, Evaluator
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)  # Single process optimization for training
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)    # Set up training
    evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)  # Set up evaluation
    # evaluator = chainercv.extensions.SemanticSegmentationEvaluator(test_iter, model, label_names=('BG', 'NG'))
    evaluator.trigger = SNAPSHOT_TRIGGER    # Set evaluation period
    # evaluator.trigger = (1, 'epoch')
    trainer.extend(evaluator)   # Set evaluation into training

    # Weight decay
    # trainer.extend(ClassWeightUpdater(model))

    # Snapshot
    # Save a current snapshot file
    trainer.extend(extensions.snapshot(
        # filename='snapshot_iter-{.updater.iteration:06d}'),
        filename='snapshot_epoch-{.updater.epoch:04d}'),
        trigger=SNAPSHOT_TRIGGER)
    # Save the latest snapshot file
    trainer.extend(extensions.snapshot(
        filename='snapshot_latest'),
        trigger=SNAPSHOT_TRIGGER)
    # Save a current model file
    trainer.extend(extensions.snapshot_object(
        target=model,
        # filename='model_iter-{.updater.iteration:06d}'),
        filename='model_epoch-{.updater.epoch:04d}'),
        trigger=SNAPSHOT_TRIGGER)
    # Save the latest model file
    trainer.extend(extensions.snapshot_object(
        target=model,
        filename='model_latest'),
        trigger=SNAPSHOT_TRIGGER)

    # Report
    # Output a logfile
    # trainer.extend(extensions.LogReport(log_name='log.txt', trigger=(1, 'iteration')))
    trainer.extend(extensions.LogReport(log_name='log.txt', trigger=(1, 'epoch')))
    # Standard out parameters
    trainer.extend(extensions.PrintReport(
        entries=['epoch', 'iteration', 'main/loss', 'validation/main/loss', 'main/miou_error',
                 'validation/main/miou_error', 'elapsed_time'],
        # log_report=extensions.LogReport(log_name=None, trigger=(10, 'iteration'))))
        log_report=extensions.LogReport(log_name=None, trigger=(1, 'epoch'))))
    # Output a graph of loss value
    trainer.extend(extensions.PlotReport(
        y_keys=['main/loss', 'validation/main/loss'],
        x_key='iteration',
        # x_key='epoch',
        file_name='loss.png',
        postprocess=lambda fig, ax, summary: ax.semilogy(),
        marker='',
        trigger=(1, 'iteration')))
    # Output a graph of mean IoU error
    trainer.extend(extensions.PlotReport(
        y_keys=['main/miou_error', 'validation/main/miou_error'],
        x_key='iteration',
        # x_key='epoch',
        file_name='miou_error.png',
        postprocess=lambda fig, ax, summary: ax.semilogy(),
        marker='',
        trigger=(1, 'iteration')))
    # Output a graph of pixel error
    trainer.extend(extensions.PlotReport(
        y_keys=['main/pixel_error', 'validation/main/pixel_error'],
        x_key='iteration',
        # x_key='epoch',
        file_name='pixel_error.png',
        postprocess=lambda fig, ax, summary: ax.semilogy(),
        marker='',
        trigger=(1, 'iteration')))
    # Output a graph of mean class error
    trainer.extend(extensions.PlotReport(
        y_keys=['main/class_error', 'validation/main/class_error'],
        x_key='iteration',
        # x_key='epoch',
        file_name='class_error.png',
        postprocess=lambda fig, ax, summary: ax.semilogy(),
        marker='',
        trigger=(1, 'iteration')))
    # Standard out a progress bar
    trainer.extend(extensions.ProgressBar(update_interval=1))

    # resume
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)  # Read a snapshot file to restart training

    # Go !
    trainer.run()

    import time
    time.sleep(60)


if __name__ == '__main__':
    _main()
