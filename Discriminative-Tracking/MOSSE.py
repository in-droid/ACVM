"""
#NOTE: We expect that this file is located in the pytracking-toolkit-lite directory, and the utils from the previous exercices are placed in the utils folder
#NOTE: Simplified version of the MOSSE tracker (presented at the lecture)
"""

import numpy as np
import cv2


import utils.ex3_utils as utils3
import utils.ex2_utils as utils2
from utils.tracker import Tracker



def x0y0wh_to_center_wh(coordinates):
    x0, y0, w, h = coordinates
    x_center = x0 + w // 2
    y_center = y0 + h // 2
    return (x_center, y_center), (w, h)


def center_wh_to_x0y0wh(coordinates):
    (x_center, y_center), w, h = coordinates

    x0 = x_center - w // 2
    y0 = y_center - h // 2

    return x0, y0, w, h


class MOSSEParameters:
    def __init__(self):
        self.sigma = 2
        self.lambda_ = 0.1
        self.alpha = 0.2
        self.scaling_factor = 1.1


class MOSSETracker(Tracker):
    def __init__(self):
        super().__init__()
        self.H_conj = None
        self.parameters = MOSSEParameters()
        self.cosine_window = None
        self.G = None
        self.w = None
        self.h = None
        self.patch_position = None
        self.scaled_w = None
        self.scaled_h = None
        # self.scaling_factor =

    def name(self):
        return "MOSSE-Tracker_FIXED_SF_1.1"

    def scale_patch(self):
        scaling_factor = self.parameters.scaling_factor
        new_w = int(self.w * scaling_factor)
        new_h = int(self.h * scaling_factor)
        if new_w % 2 == 0:
            new_w += 1
        if new_h % 2 == 0:
            new_h += 1
        return new_w, new_h

    def preprocess_patch(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = patch.astype(np.float32)
        patch = np.log1p(patch)
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def initialize(self, image, region):
        region = [int(el) for el in region]
        center, (w, h) = x0y0wh_to_center_wh(region)

        w = w + 1 if w % 2 == 0 else w
        h = h + 1 if h % 2 == 0 else h

        self.w = w
        self.h = h
        self.patch_position = list(center)

        self.scaled_w, self.scaled_h = self.scale_patch()

        patch, _ = utils2.get_patch(image, center, (self.scaled_w, self.scaled_h))
        patch = self.preprocess_patch(patch)

        self.cosine_window = utils3.create_cosine_window((self.scaled_w, self.scaled_h))
        patch = patch * self.cosine_window

        g = utils3.create_gauss_peak((self.scaled_w, self.scaled_h), self.parameters.sigma)
        self.G = np.fft.fft2(g)

        P = np.fft.fft2(patch)

        self.H_conj = ((self.G * np.conj(P)) /
                       (P * np.conj(P) + self.parameters.lambda_))

    def track(self, image):
        patch, _ = utils2.get_patch(image,
                                    self.patch_position,
                                    (self.scaled_w, self.scaled_h))
        patch = self.preprocess_patch(patch)
        L = np.fft.fft2(patch * self.cosine_window)

        g_prime = np.fft.ifft2((self.H_conj * L))

        y, x = np.unravel_index(np.argmax(g_prime),
                                g_prime.shape)

        if x > self.scaled_w // 2:
            x -= self.scaled_w

        if y > self.scaled_h // 2:
            y -= self.scaled_h

        self.patch_position[0] += x
        self.patch_position[1] += y

        patch, _ = utils2.get_patch(image, self.patch_position, (self.scaled_w, self.scaled_h))
        patch = self.preprocess_patch(patch)
        F = np.fft.fft2(patch * self.cosine_window)
        H_conj_new = (np.conj(F) * self.G) / (F * np.conj(F) + self.parameters.lambda_)
        self.H_conj = (1 - self.parameters.alpha) * self.H_conj + self.parameters.alpha * H_conj_new

        return center_wh_to_x0y0wh((self.patch_position, self.w, self.h))

