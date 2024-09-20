import numpy as np
from ex2_utils import *


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


class MSParams:
    def __init__(self):
        self.n_bins = 16
        self.sigma = 2
        self.bandwidth = 21
        self.alpha = 0
        self.eps = 1e-3


class MeanShiftTracker(Tracker):

    def __init__(self, params):
        super().__init__(params)
        self.kernel = None
        self.q = None
        self.patch_position = None
        self.mode_position = None
        self.w = None
        self.h = None

    def initialize(self, image, region):
        region = [int(el) for el in region]
        center, (w, h) = x0y0wh_to_center_wh(region)

        w = w + 1 if w % 2 == 0 else w
        h = h + 1 if h % 2 == 0 else h

        self.w = w
        self.h = h
        self.patch_position = list(center)

        patch, _ = get_patch(image, self.patch_position, (w, h))
        self.kernel = create_epanechnik_kernel(w, h, self.parameters.sigma)
        q = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        self.q = q / np.sum(q)
        self.mode_position = [w // 2, h // 2]

    def track(self, image):
        patch, _ = get_patch(image,
                             self.patch_position,
                             (self.w, self.h)
                             )
        p = extract_histogram(patch, self.parameters.n_bins, self.kernel)
        p = p / np.sum(p)

        V = np.sqrt(self.q / (p + self.parameters.eps))
        W = backproject_histogram(patch, V, self.parameters.n_bins)
        self.mode_position = mean_shift(W,
                                        self.mode_position,
                                        self.parameters.bandwidth,
                                        kernel_type='uniform',
                                        padding='constant'
                                        )

        self.patch_position[0] += self.mode_position[0] - self.w // 2
        self.patch_position[1] += self.mode_position[1] - self.h // 2

        new_patch, _ = get_patch(image, self.patch_position,
                                 (self.kernel.shape[1], self.kernel.shape[0]))
        new_q = extract_histogram(new_patch, self.parameters.n_bins, self.kernel)
        new_q = new_q / np.sum(new_q)

        self.q = (1 - self.parameters.alpha) * self.q + self.parameters.alpha * new_q
        self.q = self.q / np.sum(self.q)

        return center_wh_to_x0y0wh((self.patch_position,
                                    self.w, self.h))


def mean_shift(f,
               start_position,
               bandwidth,
               max_iter=30,
               kernel_type='uniform',
               padding='constant',
               return_path=False):

    n = bandwidth // 2
    iteratons = 0

    if padding != 'none':
        f = np.pad(f, n, mode=padding)

    coordinates = np.indices((bandwidth, bandwidth)) - n
    coordinates = coordinates[::-1]
    if padding != 'none':
        start_position = (start_position[0] + n, start_position[1] + n)

    positions = [np.array(start_position)]

    while True:
        position = positions[-1]
        position_x, position_y = position

        lower_bound_y = position_y - n
        upper_bound_y = position_y + n + 1
        lower_bound_x = position_x - n
        upper_bound_x = position_x + n + 1

        w = f[lower_bound_y: upper_bound_y, lower_bound_x: upper_bound_x]

        if np.all(w == 0):
            break

        if kernel_type == 'uniform':
            w_g = w / w.sum()
        elif kernel_type == 'normal':
            # Compute Gaussian kernel weights
            sigma = ((n - 0.5) / 3) + 1
            kernel = np.exp(-(coordinates[0] ** 2 + coordinates[1] ** 2) / (2 * sigma ** 2))
            w_g = w * kernel
            w_g /= np.sum(w_g)

        top_part = np.sum(w_g * coordinates, axis=(1, 2))

        m = top_part / w_g.sum()
        m = np.rint(m).astype(int)

        if np.linalg.norm(m) == 0:
            break

        new_position = position + m

        iteratons += 1
        if iteratons >= max_iter:
            break

        if return_path:
            positions.append(np.clip(new_position, (n, n), (f.shape[1] - n - 1, f.shape[0] - n - 1)))
        else:
            positions = [new_position]

    if return_path:
        if padding != 'none':
            return [pos - n for pos in positions]
        else:
            return [pos for pos in positions]
    if padding != 'none':
        return positions[-1] - n
    return positions[-1]


if __name__ == '__main__':
    mean_shift(None, (0, 0), 9, 20)
