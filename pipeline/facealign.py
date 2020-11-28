from __future__ import print_function
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import cv2 as cv
from PIL import Image
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


from tool import timer
from facebase import FaceBase

REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
], np.float32)

REFERENCE_X = 96
REFERENCE_Y = 112
REFERENCE_RATIO = REFERENCE_X / REFERENCE_Y


def findNonreflectiveSimilarity(uv, xy, K=2):
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=-1)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    T = T[:, 0:2].T
    return T


class FaceAlign(FaceBase):
    def __init__(self, weight_path=None, cuda=False, *, borderValue=(0, 0, 0), target_size=112):
        super().__init__(weight_path, cuda)
        self._borderValue = borderValue
        self._target_size = target_size
        self.initialized = True

    @timer
    def inference(self, image, faces, landmarks):
        assert len(faces) == len(landmarks)
        aligned_faces = []
        if isinstance(image, np.ndarray):
            pass
        elif isinstance(image, Image.Image):
            pass

        for face, landmark in zip(faces, landmarks):
            h, w, _ = face.shape
            if w / h >= REFERENCE_RATIO:
                nh = h
                nw = int(h / REFERENCE_Y * REFERENCE_X)
            else:
                nw = w
                nh = int(w / REFERENCE_X * REFERENCE_Y)

            gap_x = (w - nw) // 2
            gap_y = (h - nh) // 2

            # empty_img = np.zeros((bbox[3], bbox[2], 3), np.uint8)
            REFERENCE_FACIAL_POINTS_SCALE = []
            for i in REFERENCE_FACIAL_POINTS:
                x = i[0]
                y = i[1]
                xx = nw / REFERENCE_X * x + gap_x
                yy = nh / REFERENCE_Y * y + gap_y
                # cv.circle(empty_img, (int(xx), int(yy)), 1, (0, 0, 255), 4)
                REFERENCE_FACIAL_POINTS_SCALE.append([xx, yy])

            REFERENCE_FACIAL_POINTS_SCALE = np.array(REFERENCE_FACIAL_POINTS_SCALE, np.float32)
            KEY_POINT = np.array([landmark], np.float32)
            KEY_POINT = np.resize(KEY_POINT, [5, 2])
            similar_trans_matrix = findNonreflectiveSimilarity(KEY_POINT, REFERENCE_FACIAL_POINTS_SCALE)

            aligned_face = cv.warpAffine(image.copy(), similar_trans_matrix, (w, h),
                                         borderValue=self._borderValue)

            bg = np.ones((self._target_size, self._target_size, 3), np.uint8)
            bg *= np.array(self._borderValue, dtype=np.uint8)

            if w / h >= 1:
                nw = self._target_size
                nh = int(self._target_size / w * h)

            else:
                nh = self._target_size
                nw = int(self._target_size / h * w)

            aligned_face = cv.resize(aligned_face, (nw, nh), cv.INTER_CUBIC)
            bg[(self._target_size - nh) // 2: nh + (self._target_size - nh) // 2,
                (self._target_size - nw) // 2: nw + (self._target_size - nw) // 2, :] = aligned_face

            aligned_faces.append(bg)
        return aligned_faces
