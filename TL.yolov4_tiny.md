# Traffic light detector with YOLOv4-Tiny

#### - YOLOv4 
AP (Average Precision) and FPS (Frames Per Second) in YOLOv4 have increased by 10% and 12% respectively compared to YOLOv3.
#### - YOLOv4-tiny 
the compressed version of YOLOv4


```python
import os
import glob
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
```

## 0. Configuration setting


```python
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
 raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
```

    Found GPU at: /device:GPU:0
    


```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 18222432177172917426
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 7018292839
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 6679237784931756597
    physical_device_desc: "device: 0, name: GeForce RTX 2070 SUPER, pci bus id: 0000:01:00.0, compute capability: 7.5"
    ]
    

## 1. Data preprocssing

Change dataset to fit Yolov4 dataset format<br>
- change annotation format
- split data
- train.txt
- test.txt
- tl.name
- tlDetector.data

### 1.1 Transform  annotaition format
##### yolov4's annotaition.txt format: [object-class-id] [x-center] [y-center] [width] [height]<br>

##### [Traffic Lights id ]<br>
![traffic_lights_](https://user-images.githubusercontent.com/80514503/118099348-0f0e3d80-b410-11eb-8496-8d7b712db926.png)

#####  [x-center] [y-center] [width] [height]
Ratio of total image size<br>
- 0 < x-center, y-center < 1
- 0 <= width, height <= 1
- image size : 2048x1536


```python
txt_dir="/Users/PC921/Traffic_lights/data/row/Annotations/"
modify_dir="/Users/PC921/Traffic_lights/data/row/modify/"

def txtmodify(input_file):   
    is_first_file = True 
    out_string = ''
    name=os.path.basename(input_file)
    with open(input_file, 'r', newline='', encoding='utf-8') as in_file: 
            input_lists= in_file.readlines()
            for row in input_lists:            
                input_row = row.rstrip()
                input_list = row.split("\t") 
                if int(input_list[4])%10==4:
                    odj_class='1'
                else: odj_class=str(int(input_list[4])%10)
                xcenter = ((float(input_list[0])+float(input_list[2]))/2)/2048
                ycenter = ((float(input_list[1])+float(input_list[3]))/2)/2048
                width = (float(input_list[2])-float(input_list[0]))/1536
                height = (float(input_list[3])-float(input_list[1]))/1536
                
                out_string  += odj_class+' '+str(xcenter)+' '+str(ycenter)+' '+str(width)+' '+str(height)+'\n'
    in_file.close() 
    return out_string 

def main():
    for input_file in glob.glob(os.path.join(txt_dir, '*.txt')):
        name=os.path.basename(input_file)
        OUTPUT_PATH=modify_dir+name
        print(name)
        out_file = open(OUTPUT_PATH, 'w')
        try:    
            record = txtmodify(input_file)
            out_file.write(record)
            out_file.close()
        except Exception:
            pass

if __name__ == '__main__':
    main()
       
```

    00004259.txt
    00004260.txt
    00004261.txt
    00004262.txt
    00004263.txt
    00004264.txt
    00004265.txt
    00004266.txt
    00004267.txt
    00004268.txt
    00004269.txt
    00004270.txt
    00004271.txt
    00004272.txt
    00004273.txt
    00004274.txt
    00004275.txt
    00004276.txt
    00004277.txt
    00004278.txt
    00004279.txt
    00004280.txt
    00004281.txt
    00004282.txt
    00004283.txt
    00004284.txt
    00004285.txt
    00004286.txt
    00004287.txt
    00004288.txt
    00004289.txt
    00004290.txt
    00004291.txt
    00004292.txt
    00004293.txt
    00004294.txt
    00004295.txt
    00004296.txt
    00004297.txt
    00004298.txt
    00004299.txt
    00004300.txt
    00004301.txt
    00004302.txt
    00004303.txt
    00004304.txt
    00004305.txt
    00004306.txt
    00004307.txt
    00004308.txt
    00004309.txt
    00004310.txt
    00004311.txt
    00004312.txt
    00004313.txt
    00004314.txt
    00004315.txt
    00004316.txt
    00004317.txt
    00004318.txt
    00004319.txt
    00004320.txt
    00004321.txt
    00004322.txt
    00004323.txt
    00004324.txt
    00004325.txt
    00004326.txt
    00004327.txt
    00004328.txt
    00004329.txt
    00004330.txt
    00004331.txt
    00004332.txt
    00004333.txt
    00004334.txt
    00004335.txt
    00004336.txt
    00004337.txt
    00004338.txt
    00004339.txt
    00004340.txt
    00004341.txt
    00004342.txt
    00004343.txt
    00004344.txt
    00004345.txt
    00004346.txt
    00004347.txt
    00004348.txt
    00004349.txt
    00004350.txt
    00004351.txt
    00004352.txt
    00004353.txt
    00004354.txt
    00004355.txt
    00004356.txt
    00004357.txt
    00004358.txt
    00004359.txt
    00004360.txt
    00004361.txt
    00004362.txt
    00004363.txt
    ...
    

### 1.2 Split data (0.7:0.2:0.1)

Row data is a continuous video frame. <br>
So the data was shffleed and divided into test, train, and validation data sets.


```python
def _copy_file(src_file, dst_file):
    """copy file.
    """ 
    fpath, fname = os.path.split(dst_file)
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    shutil.copyfile(src_file, dst_file)


def split_data(data_dir, train_dir, test_dir, valid_dir, ratio=[0.7, 0.2, 0.1], shuffle=True):
    """ split data to train data, test data, valid data.
        Args:
            data_dir -- data dir to to be splitted.
            train_dir, test_dir, valid_dir -- splitted dir.
            ratio -- [train_ratio, test_ratio, valid_ratio].
            shuffle -- shuffle or not.
    """
    all_img_dir = os.path.join(data_dir, "JPEGImages/")
    all_txt_dir = os.path.join(data_dir, "modify/")
    train_dir = os.path.join(train_dir, "train_data/")
    test_dir = os.path.join(test_dir, "test_data/")
    valid_dir = os.path.join(valid_dir, "valid_data/")

    all_imgs_name = os.listdir(all_img_dir)
    img_num = len(all_imgs_name)
    train_num = int(1.0*img_num*ratio[0]/sum(ratio))
    test_num = int(1.0*img_num*ratio[1]/sum(ratio))
    valid_num = img_num-train_num-test_num

    if shuffle:
        random.shuffle(all_imgs_name)
    train_imgs_name = all_imgs_name[:train_num]
    test_imgs_name = all_imgs_name[train_num:train_num+test_num]
    valid_imgs_name = all_imgs_name[-valid_num:]

    for img_name in train_imgs_name:
        img_srcfile = os.path.join(all_img_dir, img_name)
        txt_srcfile = os.path.join(all_txt_dir, img_name.split(".")[0]+".txt")
        txt_name = img_name.split(".")[0] + ".txt"

        img_dstfile = os.path.join(train_dir, img_name)
        txt_dstfile = os.path.join(train_dir, txt_name)
        _copy_file(img_srcfile, img_dstfile)
        _copy_file(txt_srcfile, txt_dstfile)

    for img_name in test_imgs_name:
        img_srcfile = os.path.join(all_img_dir, img_name)
        txt_srcfile = os.path.join(all_txt_dir, img_name.split(".")[0]+".txt")
        txt_name = img_name.split(".")[0] + ".txt"

        img_dstfile = os.path.join(test_dir, img_name)
        txt_dstfile = os.path.join(test_dir, txt_name)
        _copy_file(img_srcfile, img_dstfile)
        _copy_file(txt_srcfile, txt_dstfile)

    for img_name in valid_imgs_name:
        img_srcfile = os.path.join(all_img_dir, img_name)
        txt_srcfile = os.path.join(all_txt_dir, img_name.split(".")[0]+".txt")
        txt_name = img_name.split(".")[0] + ".txt"

        img_dstfile = os.path.join(valid_dir, img_name)
        txt_dstfile = os.path.join(valid_dir, txt_name)
        _copy_file(img_srcfile, img_dstfile)
        _copy_file(txt_srcfile, txt_dstfile)

if __name__ == "__main__":
    data_dir='/Users/PC921/Traffic_lights/data/row'
    train_dir = "/Users/PC921/Traffic_lights/data/train"
    test_dir = "/Users/PC921/Traffic_lights/data/test"
    valid_dir = "/Users/PC921/Traffic_lights/data/valid"

    print("start splitting...")
    split_data(data_dir, train_dir, test_dir, valid_dir)
    print("\n")
    print("done.")

```

    start splitting...
    
    
    done.
    

### 1.3 Make train.txt, test.txt, valid.txt

Save the path of the image.


```python
train_dir = '/Users/PC921/Traffic_lights/data/train/train_data/'
train_txt_path='/Users/PC921/Traffic_lights/darknet/custom/train.txt'
f = open(train_txt_path,'w') 
tmp = [] 
for input_file in glob.glob(os.path.join(train_dir, '*.JPG')):
        img_path=train_dir+os.path.basename(input_file)+'\n'
        f.write(img_path)
f.close()

```


```python
test_dir = '/Users/PC921/Traffic_lights/data/test/test_data/'
test_txt_path='/Users/PC921/Traffic_lights/darknet/custom/test.txt'
f = open(test_txt_path,'w') 
for input_file in glob.glob(os.path.join(test_dir, '*.JPG')):
        img_path=test_dir+os.path.basename(input_file)+'\n'
        f.write(img_path)
f.close()
```


```python
vaild_dir = '/Users/PC921/Traffic_lights/data/valid/valid_data/'
vaild_txt_path='/Users/PC921/Traffic_lights/darknet/custom/valid.txt'
f = open(vaild_txt_path,'w') 
for input_file in glob.glob(os.path.join(vaild_dir, '*.JPG')):
        img_path=vaild_dir+os.path.basename(input_file)+'\n'
        f.write(img_path)
f.close()
```

### 1.4 tl.names


```python
class_path='/Users/PC921/Traffic_lights/darknet/custom/tl.names'
cls=["0","1","2","3","5","6","7","8","9"]
f = open(class_path,'w') 
for i in range(0,8):
    f.write(cls[i]+'\n')
f.close()
```
