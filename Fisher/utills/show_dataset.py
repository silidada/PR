import matplotlib.pyplot as plt
import matplotlib.image as mping
import os
import numpy as np
img_list = []
img_dir = os.listdir("../faces_test")
img_path_list = []
for n in img_dir:
    image_name = os.listdir(os.path.join("../faces_test",n))[0]
    img_path_list.append(os.path.join("../faces_test", n, image_name))

for n in img_path_list:
    img_list.append(mping.imread(n))

m=2
n=5
img_temp = []
for i in range(0,m*n,n):
    img_temp.append(np.concatenate(img_list[i:i+n],axis=1))
img_end = np.concatenate(img_temp,axis=0)

mping.imsave(f"../imgs/dataset.png",img_end)