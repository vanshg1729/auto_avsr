# This file is used to create a random train-val-test split for speakers data
import json
import random
import os
import math
import numpy as np
import copy

data_dir = "/ssd_scratch/cvit/vanshg/datasets/accented_speakers"
speaker_name = "diane_jennings"
speaker_dir = os.path.join(data_dir, f"{speaker_name}")

label_filepath = os.path.join(speaker_dir, f"all_labels.txt")
label_dir = os.path.dirname(label_filepath)
print(label_dir)

label_lines = open(label_filepath, 'r').readlines()
print(label_lines[0])

rng = random.Random(42)

N = len(label_lines)
test_size = 0.15
val_size = 0.10
train_size = 1 - test_size - val_size
train_label_filepath = os.path.join(label_dir, "train_labels.txt")
val_label_filepath = os.path.join(label_dir, "val_labels.txt")
test_label_filepath = os.path.join(label_dir, "test_labels.txt")

num_test = math.ceil(test_size * N)
num_val = math.ceil(val_size * N)
num_train = N - num_test - num_val
print(f"{num_test = } | {num_val = } | {num_train = }")

rng.shuffle(label_lines)

# Creating the Test label file
with open(test_label_filepath, 'w') as file:
    for i in range(num_test):
        # print(f"{i}, {label_lines[i]}", end='')
        file.write(label_lines[i])

# Creating the val label file
with open(val_label_filepath, 'w') as file:
    for i in range(num_test, num_test + num_val):
        # print(f"{i}, {label_lines[i]}", end='')
        file.write(label_lines[i])

# Creating the Train label file
with open(train_label_filepath, 'w') as file:
    for i in range(num_test + num_val, N):
        # print(f"{i}, {label_lines[i]}", end='')
        file.write(label_lines[i])
