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

import numpy as np
import os
import sys
from pathlib import Path
from utils import RawUtils
from tqdm import tqdm
from benchmark import BenchmarkLoader, RawMeta
import skimage.metrics
from typing import Tuple

def run_benchmark(bm_loader: BenchmarkLoader):
    PSNRs, SSIMs = [], []

    bar = tqdm(bm_loader)
    for input_bayer, gt_bayer, output_bayer, meta in bar:
        bar.set_description(meta.name)
        assert meta.bayer_pattern == 'BGGR'
        input_bayer, gt_bayer = RawUtils.bggr2rggb(input_bayer, gt_bayer)
       
        inp_rgb, pred_rgb, gt_rgb = RawUtils.bayer2rgb(
            input_bayer, output_bayer, gt_bayer,
            wb_gain=meta.wb_gain, CCM=meta.CCM,
        )
        inp_rgb, pred_rgb, gt_rgb = RawUtils.bggr2rggb(inp_rgb, pred_rgb, gt_rgb) # bayer-pattern unification
        bar.set_description(meta.name+'bayer_unif')

        psnrs = []
        ssims = []

        for x0, y0, x1, y1 in meta.ROIs:
            pred_patch = pred_rgb[y0:y1, x0:x1]
            gt_patch = gt_rgb[y0:y1, x0:x1]

            psnr = skimage.metrics.peak_signal_noise_ratio(gt_patch, pred_patch)
            ssim = skimage.metrics.structural_similarity(gt_patch, pred_patch, multichannel=True)
            print("psnr: ", psnr, ", ssim: ", ssim)
            psnrs.append(float(psnr))
            ssims.append(float(ssim))

        bar.set_description(meta.name+'psnr')

        PSNRs = PSNRs + psnrs   # list append
        SSIMs = SSIMs + ssims

    mean_psnr = np.mean(PSNRs)
    mean_ssim = np.mean(SSIMs)
    print("mean PSNR:", mean_psnr)
    print("mean SSIM:", mean_ssim)
    


def main():
    str_path = "./benchmark.json"
    path = Path(str_path)

    bm_loader = BenchmarkLoader(path.resolve())
    run_benchmark(bm_loader)


if __name__ == "__main__":
    main()
