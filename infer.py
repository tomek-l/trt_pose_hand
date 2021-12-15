# export OPENBLAS_CORETYPE=ARMV8; export LD_PRELOAD=/home/nx/.local/lib/python3.6/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0; export DISPLAY=:0;

import json
import math
import os
import time

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

from camera import CameraThread
from gesture_classifier import gesture_classifier
from preprocessdata import preprocessdata

DEVICE = torch.device("cuda:0")

WIDTH = 224
HEIGHT = 224

MEAN = torch.Tensor([0.485, 0.456, 0.406]).to(DEVICE)
STD = torch.Tensor([0.229, 0.224, 0.225]).to(DEVICE)


def arr2batch(arr):
    # Takes: uint8 np array, e.g. (224,224,3)
    # Outputs: torch float tensor, e.g. (1,3,224,224)

    # Equivalent to the code below
    # (which for some reason this is painfully slow):
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )

    tensor_in = torch.from_numpy(arr).to(DEVICE)
    tensor_in = tensor_in / 255.0
    tensor_in = tensor_in - MEAN
    tensor_in = tensor_in / STD
    tensor_in = tensor_in.permute(2, 0, 1)  # HWC -> CWH
    batch_in = tensor_in.unsqueeze(0)

    return batch_in


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


def classify_and_overlay(image, joints):
    dist_bn_joints = preprocessdata.find_distance(joints)
    gesture = clf.predict([dist_bn_joints, [0] * num_parts * num_parts])
    gesture_joints = gesture[0]
    preprocessdata.prev_queue.append(gesture_joints)
    preprocessdata.prev_queue.pop(0)
    preprocessdata.print_label(image, preprocessdata.prev_queue, gesture_type)


def make_trt_model():

    # Read cached model if available
    PATH_TRT = "model/hand_pose_resnet18_att_244_244_trt.pth"
    PATH_PT = "model/hand_pose_resnet18_att_244_244.pth"

    if not os.path.exists(PATH_TRT):
        print("Converting model to TRT...")
        dummy_data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

        model = (
            trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links)
            .cuda()
            .eval()
        )
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

    # Info about output vector mapping to finger joints
    with open("preprocess/hand_pose.json", "r") as f:
        hand_pose = json.load(f)

    cam = CameraThread(0, shape_in=(1920, 1080), shape_out=(224, 224))

    topology = trt_pose.coco.coco_category_to_topology(hand_pose)
    num_parts = len(hand_pose["keypoints"])
    num_links = len(hand_pose["skeleton"])

    preprocessdata = preprocessdata(topology, num_parts)
    gesture_classifier = gesture_classifier()

    # Make praser and vis. funcs
    parse_objects = ParseObjects(topology, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects = DrawObjects(topology)

    model_trt = make_trt_model().to(DEVICE).eval()

    import pickle

    filename = "svmmodel.sav"
    clf = pickle.load(open(filename, "rb"))

    with open("preprocess/gesture.json", "r") as f:
        gesture = json.load(f)
    gesture_type = gesture["classes"]

    try:
        while True:
            # Start time
            t0 = time.perf_counter()

            # Grab Frame
            arr = cam.image
            # arr = cv2.imread("merkel.jpg")
            # arr = cv2.resize(arr, (224,224))
            t1 = time.perf_counter()

            # Alternative, this takes 30ms or less:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            batch_in = arr2batch(arr)
            t2 = time.perf_counter()

            # Inference
            cmap, paf = model_trt(batch_in)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            t3 = time.perf_counter()

            # Joints
            counts, objects, peaks = parse_objects(cmap, paf)
            joints = preprocessdata.joints_inference(arr, counts, objects, peaks)
            t4 = time.perf_counter()

            # Overlay
            classify_and_overlay(arr, joints)
            t5 = time.perf_counter()

            # Visualization
            draw_joints(arr, joints)
            cv2.imshow("joint", arr)
            cv2.waitKey(1)
            t6 = time.perf_counter()

            print(f"Total                 : {1000 * (t5 - t0):0.2f}ms")
            print(f"Image copy            : {1000 * (t1 - t0):0.2f}ms")
            print(f"Preprocessing         : {1000 * (t2 - t1):0.2f}ms")
            print(f"Infer + CPU           : {1000 * (t3 - t2):0.2f}ms")
            print(f"Parsing + joint infer : {1000 * (t4 - t3):0.2f}ms")
            print(f"Gesture classification: {1000 * (t5 - t4):0.2f}ms")
            print(f"Visualization         : {1000 * (t6 - t5):0.2f}ms")
            print(f"")
    finally:
        cam.stop()
