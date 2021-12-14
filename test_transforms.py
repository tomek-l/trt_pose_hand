import torch
import torchvision.transforms as transforms
import numpy as np
import time


MEAN = torch.Tensor([0.485, 0.456, 0.406]).cuda()
STD = torch.Tensor([0.229, 0.224, 0.225]).cuda()

transform=transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])
    ]
)



if __name__ == "__main__":

    arr = np.random.random((224,224,3)) * 255
    arr = arr.astype(np.uint8)
    ITER = 1000

    for _ in range(ITER):
        t0 = time.perf_counter()
        a = transform(arr)
        t1 = time.perf_counter()
        print(f"{1000*(t1-t0):0.2f} ms")