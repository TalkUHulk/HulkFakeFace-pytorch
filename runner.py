import cv2 as cv
import os
from tqdm import tqdm
import csv
import importlib
from PIL import ImageFont, Image, ImageDraw

from facedetect import FaceDetect
from facealign import FaceAlign
from facefeatures import FaceFeatures
from facebeauty import FaceBeauty
from faceretrieval import FaceRetrival


def extract_features(src_dir, save_dir):
    cfg = importlib.import_module("RetinaFace.data.config")
    fd = FaceDetect("./weights/Resnet50_Gender_Final.pth", cuda=True, cfg=cfg.cfg_re50)
    fa = FaceAlign(target_size=112)
    ff = FaceFeatures('./weights/arcface_resnet50_epoch_30.pth', cuda=True)

    images = []
    for root, dirs, files in os.walk(src_dir):
        if not dirs:
            for file in files:
                images.append(os.path.join(root, file))

    print("total images:", len(images))
    with open(save_dir, "w") as f:
        writer = csv.writer(f)
        for i, image in enumerate(tqdm(images, desc="Features extracting")):
            try:
                star, fname = image.split('/')[-2:]
                dst_path = os.path.join("/data/datasets/chs_stars_faces", star)
                if not os.path.exists(dst_path):
                    os.mkdir(dst_path)
                img = cv.imread(image, cv.IMREAD_COLOR)
                faces, landmarks, points = fd(img, target_size=640, top=-1, max_size=1280)
                if len(faces) != 1:
                    continue
                aligned_faces = fa(img, faces, landmarks)
                cv.imwrite(os.path.join(dst_path, fname), aligned_faces[0])
                features = ff(aligned_faces)
                writer.writerow([star, fname, features[0]])
            except Exception as e:
                print(image, e)


def face_beauty(test_image):
    cfg = importlib.import_module("RetinaFace.data.config")
    fd = FaceDetect("./weights/Resnet50_Gender_Final.pth", cuda=True, cfg=cfg.cfg_re50)
    fa = FaceAlign(target_size=224)
    fb = FaceBeauty('Beauty/weight_local/beauty_ft_epoch_50.pth', cuda=True)
    img = cv.imread(test_image, cv.IMREAD_COLOR)
    faces, landmarks = fd(img, target_size=512, top=5)
    aligned_faces = fa(img, faces, landmarks)

    score = fb(aligned_faces)


def face_query(test_image):
    cfg = importlib.import_module("RetinaFace.data.config")
    fd = FaceDetect("./weights/Resnet50_Gender_Final.pth", cuda=True, cfg=cfg.cfg_re50)
    fa = FaceAlign(target_size=112)
    ff = FaceFeatures('ArcFace/weights/arcface_resnet50_epoch_30.pth', cuda=True)
    fq = FaceRetrival()
    img = cv.imread(test_image, cv.IMREAD_COLOR)
    faces, landmarks, points = fd(img, target_size=None, top=-1)
    aligned_faces = fa(img, faces, landmarks)
    features = ff(aligned_faces)
    results = fq(features)

    for point in points:
        cv.rectangle(img, (point[0], point[1]), (point[2], point[3]), color=(0, 0, 255), thickness=2)

    f_size = 50
    fontpath = "./simsun.ttc"
    font = ImageFont.truetype(fontpath, f_size)
    img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    for point, result in zip(points, results):
        draw = ImageDraw.Draw(img_pil)
        print(result[0]['star'])
        draw.text((point[0], point[1] - f_size), result[0]['star'], font=font, fill=(255, 0, 0))

    img_pil.save("result_" + test_image.split('/')[-1])





# fd = FaceDetect("./weights/Resnet50_Gender_Final.pth", cuda=False, cfg=cfg_re50)
# fa = FaceAlign(target_size=112)
# ff = FaceFeatures('./weights/arcface_resnet50_epoch_30.pth', cuda=False)
# img = cv.imread("./images/test10.jpeg", cv.IMREAD_COLOR)
# faces, landmarks, points = fd(img, target_size=None, top=-1)
# print(landmarks)
# # print(len(points))
# aligned_faces = fa(img, faces, landmarks)
# for face in aligned_faces:
#     print(points)
#     print(face.shape)
#     cv.imshow("face", face)
#     cv.waitKey()
# print(len(aligned_faces))

if __name__ == "__main__":
    # face_query("./images/test7.jpeg")
    # extract_features("/data/datasets/chs_stars_original/chs_stars_original/", "./chs_stars_features.csv")

    # face_beauty("./images/ym.jpg")
    pass