#Copyright 2022-2023 Advanced Micro Devices Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys


'''
Calculate softmax
data: data to be calculated
size: data size
return: softamx result
'''
def CPUCalcSoftmax(data, size, scale):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i] * scale)
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    #print("Softmax = ", result)
    return result

def runClassify(runner: "Runner",img): 
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])
    softmax = np.empty(pre_output_size)
    input_fixpos = inputTensors[0].get_attr("fix_point")
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    mods = ['cat','dog']
	
    """prepare batch input/output """
    outputData = []
    inputData = []
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]

    """init input image to input buffer """
    imageRun = inputData[0]
    imageRun[0, ...] = img

    """run with batch """
    job_id = runner.execute_async(inputData,outputData)
    runner.wait(job_id)

    """softmax calculate with batch """
    """Benchmark DPU FPS performance over Vitis AI APIs execute_async() and wait() """
    """Uncomment the following code snippet to include softmax calculation for model’s end-to-end FPS evaluation """
    softmax = CPUCalcSoftmax(outputData[0][0], pre_output_size, output_scale)
    top1 = mods[np.argmax(softmax)]
    maxValue = np.amax(softmax)
    
    print("Its category is: ", top1)
    print("score: ", maxValue)

"""
 obtain dpu subgrah
"""
def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

	
def main(argv):
    g = xir.Graph.deserialize(argv[1])
    subgraphs = get_child_subgraph_dpu(g)

    assert len(subgraphs) == 1 # only one DPU kernel
    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run");
		
    """input files"""
    img=cv2.imread(argv[2])

    runClassify(dpu_runners, img)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("please input dpu file.")
    else :
        main(sys.argv)

