import pandas as pd
import numpy as np
import cv2
import os
import tqdm
import glob

def toJpg(df, label_list, dir):
    for item in label_list:
        if not os.path.isdir(dir + "/" + item):
            os.makedirs(dir + "/" + item)

    for index, row in tqdm.tqdm(enumerate(df.values)):
        label = label_list[row[0]]
        image = np.array(row[1:]).reshape(28, 28, 1).astype('float32')
        cv2.imwrite(dir + "/" + label + "/" + label + "_" + str(index) + ".jpg", image)
        
if __name__ == "__main__":

    df_dir = "fashion_mnist"
    label_list = [
        "T_shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle_Boot"
    ]

    train_df = pd.read_csv(df_dir + "/fashion-mnist_train.csv")
    test_df = pd.read_csv(df_dir + "/fashion-mnist_test.csv")

    toJpg(train_df, label_list, df_dir)
    toJpg(test_df, label_list, df_dir)