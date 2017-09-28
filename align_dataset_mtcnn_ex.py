"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face_ex as detect_face
import random
from time import sleep
import cv2


def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    if args.random_order:
        random.shuffle(dataset)
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
            if args.random_order:
                random.shuffle(cls.image_paths)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename+'.png')
            print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]

                    bounding_boxes, landmarks = detect_face.detect_face(
                        img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        landmark = landmarks
                        img_size = np.asarray(img.shape)[0:2]
                        if nrof_faces > 1:
                            bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
                            img_center = img_size / 2
                            offsets = np.vstack([(det[:, 0]+det[:, 2])/2-img_center[1],
                                                 (det[:, 1]+det[:, 3])/2-img_center[0]])
                            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0)  # some extra weight on the centering
                            det = det[index, :]
                            landmark = landmark[:, index]
                        det = np.squeeze(det)
                        landmark = np.squeeze(landmark)

                        if args.align_face_image == 'off':
                            bb = np.zeros(4, dtype=np.int32)
                            bb[0] = np.maximum(det[0] - args.margin / 2, 0)
                            bb[1] = np.maximum(det[1] - args.margin / 2, 0)
                            bb[2] = np.minimum(det[2] + args.margin / 2, img_size[1])
                            bb[3] = np.minimum(det[3] + args.margin / 2, img_size[0])
                            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                            scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                            nrof_successfully_aligned += 1
                            misc.imsave(output_filename, scaled)
                        else:
                            cv_img = cv2.imread(image_path)
                            cv_img = face_alignment(cv_img, args.image_size, landmark)
                            cv2.imwrite(output_filename, cv_img)
                            nrof_successfully_aligned += 1

                        if args.landmark_image == 'on':
                            cv_img_landmark = cv2.imread(image_path)
                            # TODO : Write marked image with landmark and bounding box
                    else:
                        print('Unable to align "%s"' % image_path)
                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def face_alignment(img, face_size, f_point):
    desired_left_eye = (0.35, 0.35) #원하는 우측 눈 센터
    desired_right_eye = (0.65, 0.35)
    right_eye_center = (f_point[0], f_point[5]) #현재 우측 눈 센터
    left_eye_center = (f_point[1], f_point[6])

    # TODO : Compute eyes center, angle and image scale
    eyesCenter = ((f_point[0]+f_point[1])/2, (f_point[5]+f_point[6])/2)
    angle = np.arctan2((f_point[6]-f_point[5]),(f_point[1]-f_point[0]))*180/np.pi
    scale = (0.3*face_size)/(np.sqrt(np.add(np.power((f_point[0]-f_point[1]),2),np.power((f_point[5]-f_point[6]),2))))
    # 원하는 눈사이이의 거리 / 실제 눈사이의 거리

    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale) # 눈의 중심, 회전각도, 크기

    tX = face_size * 0.5
    tY = face_size * desired_left_eye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    (w, h) = (face_size, face_size)
    output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return output


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order', 
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--landmark_image', type=str,
        help='Write image with bounding box and landmark of detected face.', default='off', choices=['on', 'off'])
    parser.add_argument('--align_face_image', type=str,
        help='Write image cropped aligned face patch.', default='on', choices=['on', 'off'])
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
