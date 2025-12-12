#!/usr/bin/env python
# coding: utf-8

"""
Copyright 2022-2023 Advanced Micro Devices Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import time
import sys
import queue
import argparse
import vitis_ai_library


cifar2_classes = ["automobile", "truck"]


def CPUCalcArgmax(data):
    '''
    returns index of highest value in data
    '''
    val = np.argmax(data)
    return val

def preprocess_one_image(image_path, width, height, means, scales, fixpos):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    B, G, R = cv2.split(image)
    fix_scale = 2**fixpos
    B = (B - means[0]) * (scales[0] * fix_scale)
    G = (G - means[1]) * (scales[1] * fix_scale)
    R = (R - means[2]) * (scales[2] * fix_scale)
    image = cv2.merge([B, G, R])
    image = image.astype(np.int8)
    return image

def runGraphRunner(runner, images):
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()
    input_ndim = tuple(input_tensor_buffers[0].get_tensor().dims)
    batch = input_ndim[0]
    width = input_ndim[1]
    height = input_ndim[2]
    fixpos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")

    preds = []
    size = len(images) 
    group = size // batch
    if size % batch != 0 :
        group = group + 1

    for i in range(group):
        input_data = np.asarray(input_tensor_buffers[0])
        start = batch * i 
        end = min(start + batch, size) 
       
        for j in range(start, end):
            input_data[j % batch] = images[j]

        """ run graph runner"""
        job_id = runner.execute_async(input_tensor_buffers, output_tensor_buffers)
        runner.wait(job_id)

        output_data = np.asarray(output_tensor_buffers[0])
        for j in range(start, end):
            prediction_index = CPUCalcArgmax(output_data[j % batch])
            preds.append(prediction_index)

    return preds

def app(images_dir, model_name):
    images_list=os.listdir(images_dir)
    runTotal = len(images_list)
    print('Found',len(images_list),'images - processing',runTotal,'of them')

    ''' get a list of subgraphs from the compiled model file '''
    g = xir.Graph.deserialize(model_name)
    runner = vitis_ai_library.GraphRunner.create_graph_runner(g)
    """get_inputs & get_outputs"""
    input_tensor_buffers = runner.get_inputs()
    output_tensor_buffers = runner.get_outputs()

    input_ndim = tuple(input_tensor_buffers[0].get_tensor().dims)
    batch = input_ndim[0]
    width = input_ndim[1]
    height = input_ndim[2]
    fixpos = input_tensor_buffers[0].get_tensor().get_attr("fix_point")
    means = [.0, .0, .0]
    scales = [1/255.0, 1/255.0, 1/255.0]
 
    ''' Pre Processing images '''
    print("Pre-processing ",runTotal," images")

    images = []
    preds = []
    for i in range(runTotal):
        path = os.path.join(images_dir,images_list[i])
        image = preprocess_one_image(path, width, height, means, scales, fixpos)
        images.append(image)


    ''' DPU execution '''
    print("run DPU")
    time1 = time.time()

    preds = runGraphRunner(runner, images)

    time2 = time.time()
    timetotal = time2 - time1
    fps = float(runTotal / timetotal)
    print(" ")
    print("FPS=%.2f, total frames = %.0f , time=%.4f seconds" %(fps,runTotal, timetotal))
    print(" ")

    ''' Post Processing '''
    classes = cifar2_classes
    correct = 0
    wrong = 0
    for i in range(len(preds)):
        prediction = classes[preds[i]]
        ground_truth, _ = images_list[i].split("_", 1)
        if (ground_truth==prediction):
            correct += 1
        else:
            wrong += 1
    accuracy = correct/runTotal
    print("Correct: ",correct," Wrong: ",wrong," Accuracy: ", accuracy)
    return

# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--images_dir', type=str, default='../test_images', help='Path to folder of images. Default is images')
  ap.add_argument('-m', '--model',      type=str, default='./CNN_int.xmodel', help='Path of xmodel. Default is CNN_int.xmodel')

  args = ap.parse_args()
  print("\n")
  print ('Command line options:')
  print (' --images_dir : ', args.images_dir)
  print (' --model      : ', args.model)
  print("\n")

  app(args.images_dir,args.model)


if __name__ == '__main__':
  main()
