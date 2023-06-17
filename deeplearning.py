import tensorflow as tf
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from PIL import Image
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np
import itertools as itool
import networkx as nx
from statistics import mean
from PIL import Image



path = 'data/'

img_data = []
lbl_data = []
cnt_data = []

# Run on user
for data in os.listdir(path):
    data_path = os.path.join(path, data)
    print()
    cnt = 0
    for p, file_name in enumerate(os.listdir(data_path)):

        if cnt > 215:
            continue

        if ("jpg" not in file_name) and ("JPG" not in file_name):
            continue

        try:
            file_path = os.path.join(data_path, file_name)

            img_tmp = Image.open(file_path).convert('RGB')

            img_tmp = img_tmp.resize((64,64))
            
            tmp_img = np.asarray(img_tmp)
            cnt += 1
            
            img_data.append(tmp_img)
            lbl_data.append(int(data_path[-1]))
        except:
            print("error")
            print(file_path)
        
    cnt_data.append(cnt)


d1_train, d1_test, d1_val = int(cnt_data[0]*0.6), int(cnt_data[0]*0.2), int(cnt_data[0]*0.2)
d2_train, d2_test, d2_val = int(cnt_data[1]*0.6), int(cnt_data[1]*0.2), int(cnt_data[1]*0.2)
d3_train, d3_test, d3_val = int(cnt_data[2]*0.6), int(cnt_data[2]*0.2), int(cnt_data[2]*0.2)
d4_train, d4_test, d4_val = int(cnt_data[3]*0.6), int(cnt_data[3]*0.2), int(cnt_data[3]*0.2)

d1_test += d1_train
d1_val += d1_test

d2_train += d1_val
d2_test += d2_train
d2_val += d2_test

d3_train += d2_val
d3_test += d3_train
d3_val += d3_test

d4_train += d3_val
d4_test += d4_train
d4_val += d4_test

train_img = []
train_lbl = []

test_img = []
test_lbl = []

val_img = []
val_lbl = []

# Train image
train_img.extend(img_data[:d1_train])
train_img.extend(img_data[d1_val:d2_train])
train_img.extend(img_data[d2_val:d3_train])
train_img.extend(img_data[d3_val:d4_train])

# Train label
train_lbl.extend([0 for _ in range(d1_train)])
train_lbl.extend([1 for _ in range(d1_val, d2_train)])
train_lbl.extend([2 for _ in range(d2_val, d3_train)])
train_lbl.extend([3 for _ in range(d3_val, d4_train)])



# Test image
test_img.extend(img_data[d1_train:d1_test])
test_img.extend(img_data[d2_train:d2_test])
test_img.extend(img_data[d3_train:d3_test])
test_img.extend(img_data[d4_train:d4_test])

# Test label
test_lbl.extend([0 for _ in range(d1_train, d1_test)])
test_lbl.extend([1 for _ in range(d2_train, d2_test)])
test_lbl.extend([2 for _ in range(d3_train, d3_test)])
test_lbl.extend([3 for _ in range(d4_train, d4_test)])



# Val image
val_img.extend(img_data[d1_test:d1_val])
val_img.extend(img_data[d2_test:d2_val])
val_img.extend(img_data[d3_test:d3_val])
val_img.extend(img_data[d4_test:d4_val])

# Val label
val_lbl.extend([0 for _ in range(d1_test, d1_val)])
val_lbl.extend([1 for _ in range(d2_test, d2_val)])
val_lbl.extend([2 for _ in range(d3_test, d3_val)])
val_lbl.extend([2 for _ in range(d4_test, d4_val)])


train_img = np.array(train_img)
train_lbl = np.array(train_lbl)
test_img = np.array(test_img)
test_lbl = np.array(test_lbl)
val_img = np.array(val_img)
val_lbl = np.array(val_lbl)

tf.random.set_seed(777)
# 모델 생성
model = models.Sequential()
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape = (64,64,3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))



epoch = 5
batch_size = 50

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_img, train_lbl,  
                    validation_data=(val_img, val_lbl),
                    epochs=epoch,
                    batch_size=batch_size,
                    shuffle=True)



path = 'testdata/'

img_dt = []

while 1:    
    for data in os.listdir(path):
        data_path = os.path.join(path, data)

        for p, file_name in enumerate(os.listdir(data_path)):
            if "jpg" not in file_name:
                continue

            file_path = os.path.join(data_path, file_name)
            print(file_name)

            img_tmp = Image.open(file_path)
            img_tmp = img_tmp.resize((64,64))
            tmp_img = np.asarray(img_tmp)

            
            img_dt.append(tmp_img)

    pred_img = []
    pred_img.extend(img_dt)
    pred_img = np.array(pred_img)


    predictions = model.predict(pred_img)
    predicted_labels = np.argmax(predictions, axis=1)

    predict_v = predicted_labels[-1]

    f = open("data.txt", 'w')
    f.write(str(predict_v))
    f.close()