# export OPENBLAS_CORETYPE=ARMV8; export LD_PRELOAD=/home/nx/.local/lib/python3.6/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0; export DISPLAY=:0;

import json
import math
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch2trt
import torchvision.transforms as transforms
import trt_pose.coco
import trt_pose.models
from PIL import Image
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

from gesture_classifier import gesture_classifier
from preprocessdata import preprocessdata

device = torch.device("cuda")

def preprocess(image):
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

    device = torch.device("cuda")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def draw_joints(image, joints):
    count = 0
    for i in joints:
        if i == [0, 0]:
            count += 1
    if count >= 3:
        return
    for i in joints:
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 1)
    cv2.circle(image, (joints[0][0], joints[0][1]), 2, (255, 0, 255), 1)
    for i in hand_pose["skeleton"]:
        if joints[i[0] - 1][0] == 0 or joints[i[1] - 1][0] == 0:
            break
        cv2.line(
            image,
            (joints[i[0] - 1][0], joints[i[0] - 1][1]),
            (joints[i[1] - 1][0], joints[i[1] - 1][1]),
            (0, 255, 0),
            1,
        )

def make_model():

    # Read cached model if available
    PATH_TRT = "model/hand_pose_resnet18_att_244_244_trt.pth"
    PATH_PT = "model/hand_pose_resnet18_att_244_244.pth"

    if not os.path.exists(PATH_TRT):
        print("Converting model to TRT...")
        dummy_data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
        model.load_state_dict(torch.load(PATH_PT))
        model_trt = torch2trt.torch2trt(
            model, [dummy_data], fp16_mode=True, max_workspace_size=1 << 25
        )
        torch.save(model_trt.state_dict(), PATH_TRT)

    print("Loading model...")
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(PATH_TRT))
    print("Model loaded!")

    return model_trt


if __name__ == "__main__":
    WIDTH = 224
    HEIGHT = 224

    # Info about output vector mapping to finger joints
    with open("preprocess/hand_pose.json", "r") as f:
        hand_pose = json.load(f)

    # TODO: not needed?
    topology = trt_pose.coco.coco_category_to_topology(hand_pose)
    num_parts = len(hand_pose["keypoints"])
    num_links = len(hand_pose["skeleton"])
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

    preprocessdata = preprocessdata(topology, num_parts)
    gesture_classifier = gesture_classifier()

    parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects = DrawObjects(topology)

    model_trt = make_model()

    while True:
        print("heheszki")
        img = Image.open("merkel.jpg")
        # img = Image.open('hand.jpg')
        arr = np.array(img)
        cv2.imshow("angela", arr)
        arr = cv2.resize(arr, (224, 224))

        image = arr
        data = preprocess(image)
        cmap, paf = model_trt(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        joints = preprocessdata.joints_inference(image, counts, objects, peaks)
        draw_joints(image, joints)

        cv2.imshow("joint", image)
        cv2.waitKey(1)
