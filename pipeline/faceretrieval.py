from __future__ import print_function
from sklearn.preprocessing import normalize
import csv
import pickle
import numpy as np
from milvus import Milvus

from tool import timer
from facebase import FaceBase


class FaceRetrival(FaceBase):
    def __init__(self, weight_path=None,
                 cuda=False, *,
                 _HOST='localhost',
                 _PORT='19530',
                 _collection_name='chs_stars_faces_512',  # 'chs_stars_faces',
                 _PCA=None,  # "chs_stars_features_pca.pickle",
                 _data_base="chs_stars_labels.csv",  # "chs_stars_features_pca.csv", #
                 _nProbe=64,
                 _top_k=5):
        super().__init__(weight_path, cuda)
        self._nprobe = _nProbe
        self._collection_name = _collection_name
        self._HOST = _HOST
        self._top_k = _top_k
        self._milvus = Milvus(_HOST, _PORT)
        self._data_base = {}
        self._pca_model = None
        if _PCA:
            with open(_PCA, "rb") as f:
                self._pca_model = pickle.load(f)
        with open(_data_base, "r") as f:
            reader = csv.reader(f)
            for line in reader:
                self._data_base[line[0]] = line[1:3]
        self.initialized = True

    @timer
    def inference(self, features):

        if isinstance(features, list):
            features = np.array(features)
            num = len(features)
        else:
            raise NotImplementedError()
        features = features.reshape((num, 512))
        if self._pca_model:
            features = normalize(features)
            features = self._pca_model.transform(features).squeeze().tolist()
        else:
            features = normalize(features).squeeze().tolist()
        search_param = {
            "nprobe": self._nprobe
        }
        param = {
            'collection_name': self._collection_name,
            'query_records': [features] if num == 1 else features,
            'top_k': self._top_k,
            'params': search_param,
        }

        status, q_results = self._milvus.search(**param)
        ret = []
        if status.OK():
            for id_list, dis_list in zip(q_results.id_array, q_results.distance_array):
                for id, dis in zip(id_list, dis_list):
                    tmp = self._data_base[str(id)]
                    ret.append({"star": tmp[0], "img": tmp[1], "dis": dis})

        result = [ret[i * self._top_k: (i + 1) * self._top_k] for i in range(num)]
        return result
