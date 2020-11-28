from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import csv
import json
import pickle
import numpy as np
from milvus import Milvus, IndexType, MetricType, Status
from tqdm import tqdm


# _HOST = 'localhost'
# _PORT = '19530'
# _collection_name = 'chs_stars_faces'
# _DIM = 179  # dimension of vector
# _INDEX_FILE_SIZE = 256  # max file size of stored index
#
# milvus = Milvus(_HOST, _PORT)
# param = {
#     'collection_name': _collection_name,
#     'dimension': _DIM,
#     'index_file_size': _INDEX_FILE_SIZE,  # optional
#     'metric_type': MetricType.L2  # optional
# }


def _pca():
    with open("../chs_stars_features.csv", "r") as fr:
        reader = csv.reader(fr)
        lines = []
        for i, line in enumerate(reader):
            lines.append(line)
        total = len(lines)
        print("total:", total)
        n = np.array([i for i in range(total)])
        selects = np.random.choice(n, size=10000, replace=False)
        all_embedding = []
        for index in selects:
            star, fname, features = lines[index]
            features = np.array(json.loads(features))
            embedding = np.resize(features, (1, 512))
            embedding = normalize(embedding)
            all_embedding.append(embedding.squeeze())

    all_embedding = np.array(all_embedding).astype(np.float32)
    print(all_embedding.shape)

    pca = PCA(n_components=0.95)
    pca = pca.fit(all_embedding)

    X_dr = pca.transform(all_embedding)
    print(X_dr.shape)

    with open("chs_stars_features_pca.pickle", "wb") as f:
        pickle.dump(pca, f)

    """
    total: 101500
    (10000, 512)
    (10000, 179)
    """

def create():
    _HOST = 'localhost'
    _PORT = '19530'
    _collection_name = 'chs_stars_faces_512'
    _DIM = 512  # dimension of vector
    _INDEX_FILE_SIZE = 256  # max file size of stored index

    milvus = Milvus(_HOST, _PORT)
    param = {
        'collection_name': _collection_name,
        'dimension': _DIM,
        'index_file_size': _INDEX_FILE_SIZE,  # optional
        'metric_type': MetricType.IP  # optional
    }


    milvus.create_collection(param)
    index_param = {
        'nlist': 2048  # 推荐 4 * sqrt(n)
    }

    status = milvus.create_index(_collection_name, IndexType.IVF_SQ8, index_param)

    # with open("chs_stars_features_pca.pickle", "rb") as f:
    #     pca = pickle.load(f)
    #
    # with open("../chs_stars_features_pca.csv", "w") as fw, open("../chs_stars_features.csv", "r") as fr:
    #     reader = csv.reader(fr)
    #     writer = csv.writer(fw)
    #     for index, line in enumerate(tqdm(reader)):
    #         star, fname, features = line
    #         features = np.array(json.loads(features))
    #         features = np.resize(features, (1, 512))
    #         features = normalize(features)
    #         features = pca.transform(features).squeeze()
    #         status, ids = milvus.insert(collection_name=_collection_name, records=[features.tolist()], ids=[index])
    #         if not status.OK():
    #             print(status)
    #             continue
    #         writer.writerow([index, star, fname, features])

    with open("../chs_stars_labels.csv", "w") as fw, open("../chs_stars_features.csv", "r") as fr:
        reader = csv.reader(fr)
        writer = csv.writer(fw)
        for index, line in enumerate(tqdm(reader)):
            star, fname, features = line
            # features = np.array(json.loads(features))
            # features = np.resize(features, (1, 512))
            #features = normalize(features)
            features = json.loads(features)
            status, ids = milvus.insert(collection_name=_collection_name, records=[features], ids=[index])
            if not status.OK():
                print(status)
                continue
            writer.writerow([index, star, fname])


create()