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
- images
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
                ycenter = ((float(input_list[1])+float(input_list[3]))/2)/1536
                width = (float(input_list[2])-float(input_list[0]))/2048
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
    00004364.txt
    00004365.txt
    00004366.txt
    00004367.txt
    00004368.txt
    00004369.txt
    00004370.txt
    00004371.txt
    00004372.txt
    00004373.txt
    00004374.txt
    00004375.txt
    00004376.txt
    00004377.txt
    00004378.txt
    00004379.txt
    00004380.txt
    00004381.txt
    00004382.txt
    00004383.txt
    00004384.txt
    00004385.txt
    00004386.txt
    00004387.txt
    00004388.txt
    00004389.txt
    00004390.txt
    00004391.txt
    00004392.txt
    00004393.txt
    00004394.txt
    00004395.txt
    00004396.txt
    00004397.txt
    00004398.txt
    00004399.txt
    00004400.txt
    00004401.txt
    00004402.txt
    00004403.txt
    00004404.txt
    00004405.txt
    00004406.txt
    00004407.txt
    00004408.txt
    00004409.txt
    00004410.txt
    00004411.txt
    00004412.txt
    00004413.txt
    00004414.txt
    00004415.txt
    00004416.txt
    00004417.txt
    00004418.txt
    00004419.txt
    00004420.txt
    00004421.txt
    00004422.txt
    00004423.txt
    00004424.txt
    00004425.txt
    00004426.txt
    00004427.txt
    00004428.txt
    00004429.txt
    00004430.txt
    00004431.txt
    00004432.txt
    00004433.txt
    00004434.txt
    00004435.txt
    00004436.txt
    00004437.txt
    00004438.txt
    00004439.txt
    00004440.txt
    00004441.txt
    00004442.txt
    00004443.txt
    00004444.txt
    00004445.txt
    00004446.txt
    00004447.txt
    00004448.txt
    00004449.txt
    00004450.txt
    00004451.txt
    00004452.txt
    00004453.txt
    00004454.txt
    00004455.txt
    00004456.txt
    00004457.txt
    00004458.txt
    00004459.txt
    00004460.txt
    00004461.txt
    00004462.txt
    00004463.txt
    00004464.txt
    00004465.txt
    00004466.txt
    00004467.txt
    00004468.txt
    00004469.txt
    00004470.txt
    00004471.txt
    00004472.txt
    00004473.txt
    00004474.txt
    00004475.txt
    00004476.txt
    00004477.txt
    00004478.txt
    00004479.txt
    00004480.txt
    00004481.txt
    00004482.txt
    00004483.txt
    00004484.txt
    00004485.txt
    00004486.txt
    00004487.txt
    00004488.txt
    00004489.txt
    00004490.txt
    00004491.txt
    00004492.txt
    00004493.txt
    00004494.txt
    00004495.txt
    00004496.txt
    00004497.txt
    00004498.txt
    00004499.txt
    00004500.txt
    00004501.txt
    00004502.txt
    00004503.txt
    00004504.txt
    00004505.txt
    00004506.txt
    00004507.txt
    00004508.txt
    00004509.txt
    00004510.txt
    00004511.txt
    00004512.txt
    00004513.txt
    00004514.txt
    00004515.txt
    00004516.txt
    00004517.txt
    00004518.txt
    00004519.txt
    00004520.txt
    00004521.txt
    00004522.txt
    00004523.txt
    00004524.txt
    00004525.txt
    00004526.txt
    00004527.txt
    00004528.txt
    00004529.txt
    00004530.txt
    00004531.txt
    00004532.txt
    00004533.txt
    00004534.txt
    00004535.txt
    00004536.txt
    00004537.txt
    00004538.txt
    00004539.txt
    00004540.txt
    00004541.txt
    00004542.txt
    00004543.txt
    00004544.txt
    00004545.txt
    00004546.txt
    00004547.txt
    00004548.txt
    00004549.txt
    00004550.txt
    00004551.txt
    00004552.txt
    00004553.txt
    00004554.txt
    00004555.txt
    00004556.txt
    00004557.txt
    00004558.txt
    00004559.txt
    00004560.txt
    00004561.txt
    00004562.txt
    00004563.txt
    00004564.txt
    00004565.txt
    00004566.txt
    00004567.txt
    00004568.txt
    00004569.txt
    00004570.txt
    00004571.txt
    00004572.txt
    00004573.txt
    00004574.txt
    00004575.txt
    00004576.txt
    00004577.txt
    00004578.txt
    00004579.txt
    00004580.txt
    00004581.txt
    00004582.txt
    00004583.txt
    00004584.txt
    00004585.txt
    00004586.txt
    00004587.txt
    00004588.txt
    00004589.txt
    00004590.txt
    00004591.txt
    00004592.txt
    00004593.txt
    00004594.txt
    00004595.txt
    00004596.txt
    00004597.txt
    00004598.txt
    00004599.txt
    00004600.txt
    00004601.txt
    00004602.txt
    00004603.txt
    00004604.txt
    00004605.txt
    00004606.txt
    00004607.txt
    00004608.txt
    00004609.txt
    00004610.txt
    00004611.txt
    00004612.txt
    00004613.txt
    00004614.txt
    00004615.txt
    00004616.txt
    00004617.txt
    00004618.txt
    00004619.txt
    00004620.txt
    00004621.txt
    00004622.txt
    00004623.txt
    00004624.txt
    00004625.txt
    00004626.txt
    00004627.txt
    00004628.txt
    00004629.txt
    00004630.txt
    00004631.txt
    00004632.txt
    00004633.txt
    00004634.txt
    00004635.txt
    00004636.txt
    00004637.txt
    00004638.txt
    00004639.txt
    00004640.txt
    00004641.txt
    00004642.txt
    00004643.txt
    00004644.txt
    00004645.txt
    00004646.txt
    00004647.txt
    00004648.txt
    00004649.txt
    00004650.txt
    00004651.txt
    00004652.txt
    00004653.txt
    00004654.txt
    00004655.txt
    00004656.txt
    00004657.txt
    00004658.txt
    00004659.txt
    00004660.txt
    00004661.txt
    00004662.txt
    00004663.txt
    00004664.txt
    00004665.txt
    00004666.txt
    00004667.txt
    00004668.txt
    00004669.txt
    00004670.txt
    00004671.txt
    00004672.txt
    00004673.txt
    00004674.txt
    00004675.txt
    00004676.txt
    00004677.txt
    00004678.txt
    00004679.txt
    00004680.txt
    00004681.txt
    00004682.txt
    00004683.txt
    00004684.txt
    00004685.txt
    00004686.txt
    00004687.txt
    00004688.txt
    00004689.txt
    00004690.txt
    00004691.txt
    00004692.txt
    00004693.txt
    00004694.txt
    00004695.txt
    00004696.txt
    00004697.txt
    00004698.txt
    00004699.txt
    00004700.txt
    00004701.txt
    00004702.txt
    00004703.txt
    00004704.txt
    00004705.txt
    00004706.txt
    00004707.txt
    00004708.txt
    00004709.txt
    00004710.txt
    00004711.txt
    00004712.txt
    00004713.txt
    00004714.txt
    00004715.txt
    00004716.txt
    00004717.txt
    00004718.txt
    00004719.txt
    00004720.txt
    00004721.txt
    00004722.txt
    00004723.txt
    00004724.txt
    00004725.txt
    00004726.txt
    00004727.txt
    00004728.txt
    00004729.txt
    00004730.txt
    00004731.txt
    00004732.txt
    00004733.txt
    00004734.txt
    00004735.txt
    00004736.txt
    00004737.txt
    00004738.txt
    00004739.txt
    00004740.txt
    00004741.txt
    00004742.txt
    00004743.txt
    00004744.txt
    00004745.txt
    00004746.txt
    00004747.txt
    00004748.txt
    00004749.txt
    00004750.txt
    00004751.txt
    00004752.txt
    00004753.txt
    00004754.txt
    00004755.txt
    00004756.txt
    00004757.txt
    00004758.txt
    00004759.txt
    00004760.txt
    00004761.txt
    00004762.txt
    00004763.txt
    00004764.txt
    00004765.txt
    00004766.txt
    00004767.txt
    00004768.txt
    00004769.txt
    00004770.txt
    00004771.txt
    00004772.txt
    00004773.txt
    00004774.txt
    00004775.txt
    00004776.txt
    00004777.txt
    00004778.txt
    00004779.txt
    00004780.txt
    00004781.txt
    00004782.txt
    00004783.txt
    00004784.txt
    00004785.txt
    00004786.txt
    00004787.txt
    00004788.txt
    00004789.txt
    00004790.txt
    00004791.txt
    00004792.txt
    00004793.txt
    00004794.txt
    00004795.txt
    00004796.txt
    00004797.txt
    00004798.txt
    00004799.txt
    00004800.txt
    00004801.txt
    00004802.txt
    00004803.txt
    00004804.txt
    00004805.txt
    00004806.txt
    00004807.txt
    00004808.txt
    00004809.txt
    00004810.txt
    00004811.txt
    00004812.txt
    00004813.txt
    00004814.txt
    00004815.txt
    00004816.txt
    00004817.txt
    00004818.txt
    00004819.txt
    00004820.txt
    00004821.txt
    00004822.txt
    00004823.txt
    00004824.txt
    00004825.txt
    00004826.txt
    00004827.txt
    00004828.txt
    00004829.txt
    00004830.txt
    00004831.txt
    00004832.txt
    00004833.txt
    00004834.txt
    00004835.txt
    00004836.txt
    00004837.txt
    00004838.txt
    00004839.txt
    00004840.txt
    00004841.txt
    00004842.txt
    00004843.txt
    00004844.txt
    00004845.txt
    00004846.txt
    00004847.txt
    00004848.txt
    00004849.txt
    00004850.txt
    00004851.txt
    00004852.txt
    00004853.txt
    00004854.txt
    00004855.txt
    00004856.txt
    00004857.txt
    00004858.txt
    00004859.txt
    00004860.txt
    00004861.txt
    00004862.txt
    00004863.txt
    00004864.txt
    00004865.txt
    00004866.txt
    00004867.txt
    00004868.txt
    00004869.txt
    00004870.txt
    00004871.txt
    00004872.txt
    00004873.txt
    00004874.txt
    00004875.txt
    00004876.txt
    00004877.txt
    00004878.txt
    00004879.txt
    00004880.txt
    00004881.txt
    00004882.txt
    00004883.txt
    00004884.txt
    00004885.txt
    00004886.txt
    00004887.txt
    00004888.txt
    00004889.txt
    00004890.txt
    00004891.txt
    00004892.txt
    00004893.txt
    00004894.txt
    00004895.txt
    00004896.txt
    00004897.txt
    00004898.txt
    00004899.txt
    00004900.txt
    00004901.txt
    00004902.txt
    00004903.txt
    00004904.txt
    00004905.txt
    00004906.txt
    00004907.txt
    00004908.txt
    00004909.txt
    00004910.txt
    00004911.txt
    00004912.txt
    00004913.txt
    00004914.txt
    00004915.txt
    00004916.txt
    00004917.txt
    00004918.txt
    00004919.txt
    00004920.txt
    00004921.txt
    00004922.txt
    00004923.txt
    00004924.txt
    00004925.txt
    00004926.txt
    00004927.txt
    00004928.txt
    00004929.txt
    00004930.txt
    00004931.txt
    00004932.txt
    00004933.txt
    00004934.txt
    00004935.txt
    00004936.txt
    00004937.txt
    00004938.txt
    00004939.txt
    00004940.txt
    00004941.txt
    00004942.txt
    00004943.txt
    00004944.txt
    00004945.txt
    00004946.txt
    00004947.txt
    00004948.txt
    00004949.txt
    00004950.txt
    00004951.txt
    00004952.txt
    00004953.txt
    00004954.txt
    00004955.txt
    00004956.txt
    00004957.txt
    00004958.txt
    00004959.txt
    00004960.txt
    00004961.txt
    00004962.txt
    00004963.txt
    00004964.txt
    00004965.txt
    00004966.txt
    00004967.txt
    00004968.txt
    00004969.txt
    00004970.txt
    00004971.txt
    00004972.txt
    00004973.txt
    00004974.txt
    00004975.txt
    00004976.txt
    00004977.txt
    00004978.txt
    00004979.txt
    00004980.txt
    00004981.txt
    00004982.txt
    00004983.txt
    00004984.txt
    00004985.txt
    00004986.txt
    00004987.txt
    00004988.txt
    00004989.txt
    00004990.txt
    00004991.txt
    00004992.txt
    00004993.txt
    00004994.txt
    00004995.txt
    00004996.txt
    00004997.txt
    00004998.txt
    00004999.txt
    00005000.txt
    00005001.txt
    00005002.txt
    00005003.txt
    00005004.txt
    00005005.txt
    00005006.txt
    00005007.txt
    00005008.txt
    00005009.txt
    00005010.txt
    00005011.txt
    00005012.txt
    00005013.txt
    00005014.txt
    00005015.txt
    00005016.txt
    00005017.txt
    00005018.txt
    00005019.txt
    00005020.txt
    00005021.txt
    00005022.txt
    00005023.txt
    00005024.txt
    00005025.txt
    00005026.txt
    00005027.txt
    00005028.txt
    00005029.txt
    00005030.txt
    00005031.txt
    00005032.txt
    00005033.txt
    00005034.txt
    00005035.txt
    00005036.txt
    00005037.txt
    00005038.txt
    00005039.txt
    00005040.txt
    00005041.txt
    00005042.txt
    00005043.txt
    00005044.txt
    00005045.txt
    00005046.txt
    00005047.txt
    00005048.txt
    00005049.txt
    00005050.txt
    00005051.txt
    00005052.txt
    00005053.txt
    00005054.txt
    00005055.txt
    00005056.txt
    00005057.txt
    00005058.txt
    00005059.txt
    00005060.txt
    00005061.txt
    00005062.txt
    00005063.txt
    00005064.txt
    00005065.txt
    00005066.txt
    00005067.txt
    00005068.txt
    00005069.txt
    00005070.txt
    00005071.txt
    00005072.txt
    00005073.txt
    00005074.txt
    00005075.txt
    00005076.txt
    00005077.txt
    00005078.txt
    00005079.txt
    00005080.txt
    00005081.txt
    00005082.txt
    00005083.txt
    00005084.txt
    00005085.txt
    00005086.txt
    00005087.txt
    00005088.txt
    00005089.txt
    00005090.txt
    00005091.txt
    00005092.txt
    00005093.txt
    00005094.txt
    00005095.txt
    00005096.txt
    00005097.txt
    00005098.txt
    00005099.txt
    00005100.txt
    00005101.txt
    00005102.txt
    00005103.txt
    00005104.txt
    00005105.txt

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
test_txt_path='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/custom/test.txt'
f = open(test_txt_path,'w') 
for input_file in glob.glob(os.path.join(test_dir, '*.JPG')):
        img_path=test_dir+os.path.basename(input_file)+'\n'
        f.write(img_path)
f.close()
```


```python
vaild_dir = '/Users/PC921/Traffic_lights/data/valid/valid_data/'
vaild_txt_path='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/custom/valid.txt'
f = open(vaild_txt_path,'w') 
for input_file in glob.glob(os.path.join(vaild_dir, '*.JPG')):
        img_path=vaild_dir+os.path.basename(input_file)+'\n'
        f.write(img_path)
f.close()
```

### 1.4 tl.names


```python
class_path='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/custom/tl.names'
cls=["0","1","2","3","5","6","7","8","9"]
f = open(class_path,'w') 
for i in range(0,9):
    f.write(cls[i]+'\n')
f.close()
```

### 1.5 tl.data


```python
class_path='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/custom/tl.data'
f = open(class_path,'w') 

f.write('classes= 9'+'\n')
f.write('train = custom/train.txt '+'\n')
f.write('valid = custom/valid.txt'+'\n')
f.write('test = custom/test.txt'+'\n')
f.write('names = custom/tl.names'+'\n')
f.write('backup = backup/'+'\n')
f.close()

```

## 3. Train custom data


```python
%cd /Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/

```

    C:\Users\PC921\Traffic_lights\vcpkg\installed\x64-windows\tools\darknet
    


```python
!./darknet detector train custom/tl.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map
```

## 4. Test demo

1) testdemo1


```python
! darknet detector demo custom/tl.data cfg/yolov4-tiny-custom.cfg  backup/yolov4-tiny-custom_best.weights -dont_show testdemo.mp4 -i 0 -out_filename testdemo1_result.avi
```

     CUDNN_HALF=1 
    Demo
    net.optimized_memory = 0 
    mini_batch = 1, batch = 16, time_steps = 1, train = 0 
    nms_kind: greedynms (1), beta = 0.600000 
    nms_kind: greedynms (1), beta = 0.600000 
    
     seen 64, trained: 1070 K-images (16 Kilo-batches_64) 
    video file: testdemo.mp4
    Video stream: 854 x 480 
    Objects:
    
    
    FPS:0.0 	 AVG_FPS:0.0
    Objects:
    
    0: 99% 
    0: 99% 
    
    FPS:7.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 99% 
    
    FPS:28.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 99% 
    
    FPS:35.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 98% 
    
    FPS:38.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 98% 
    
    FPS:41.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 98% 
    
    FPS:43.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 98% 
    
    FPS:46.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 97% 
    
    FPS:48.1 	 AVG_FPS:0.0
    
    

     CUDA-version: 10020 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     OpenCV version: 4.3.0
     0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070 SUPER 
       layer   filters  size/strd(dil)      input                output
       0 conv     32       3 x 3/ 2    416 x 416 x   3 ->  208 x 208 x  32 0.075 BF
       1 conv     64       3 x 3/ 2    208 x 208 x  32 ->  104 x 104 x  64 0.399 BF
       2 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
       3 route  2 		                       1/2 ->  104 x 104 x  32 
       4 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       5 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       6 route  5 4 	                           ->  104 x 104 x  64 
       7 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
       8 route  2 7 	                           ->  104 x 104 x 128 
       9 max                2x 2/ 2    104 x 104 x 128 ->   52 x  52 x 128 0.001 BF
      10 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      11 route  10 		                       1/2 ->   52 x  52 x  64 
      12 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      13 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      14 route  13 12 	                           ->   52 x  52 x 128 
      15 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      16 route  10 15 	                           ->   52 x  52 x 256 
      17 max                2x 2/ 2     52 x  52 x 256 ->   26 x  26 x 256 0.001 BF
      18 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      19 route  18 		                       1/2 ->   26 x  26 x 128 
      20 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      21 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      22 route  21 20 	                           ->   26 x  26 x 256 
      23 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      24 route  18 23 	                           ->   26 x  26 x 512 
      25 max                2x 2/ 2     26 x  26 x 512 ->   13 x  13 x 512 0.000 BF
      26 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      27 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
      28 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
      29 conv     42       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  42 0.007 BF
      30 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
      31 route  27 		                           ->   13 x  13 x 256 
      32 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
      33 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
      34 route  33 23 	                           ->   26 x  26 x 384 
      35 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
      36 conv     42       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  42 0.015 BF
      37 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    Total BFLOPS 6.800 
    avg_outputs = 300731 
     Allocate additional workspace_size = 26.22 MB 
    Loading weights from backup/yolov4-tiny-custom_best.weights...Done! Loaded 38 layers from weights-file 
    

     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 97% 
    
    FPS:50.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 97% 
    
    FPS:51.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 96% 
    
    FPS:53.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 96% 
    
    FPS:54.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 96% 
    
    FPS:76.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 96% 
    
    FPS:93.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    0: 95% 
    
    FPS:97.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    0: 87% 
    
    FPS:94.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    0: 87% 
    
    FPS:91.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    0: 90% 
    
    FPS:111.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    0: 88% 
    
    FPS:109.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    0: 84% 
    
    FPS:104.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 62% 
    0: 94% 
    
    FPS:122.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 62% 
    0: 94% 
    
    FPS:119.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 61% 
    0: 94% 
    
    FPS:114.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 82% 
    0: 77% 
    
    FPS:109.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 89% 
    0: 71% 
    
    FPS:105.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:123.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:121.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:115.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:133.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:128.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:122.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:111.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:113.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:108.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:100.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:97.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:94.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:91.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:88.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:86.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:84.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:83.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:87.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:85.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:83.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:81.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:80.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:79.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:96.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:112.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:126.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:123.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:117.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:112.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:121.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:134.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:144.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:160.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:163.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:169.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:161.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:152.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:143.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:136.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:151.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:144.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:137.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:147.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:141.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:157.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:150.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:154.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:159.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:169.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:159.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:149.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:163.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:156.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:169.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:181.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:187.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:191.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:156.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:169.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:161.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:156.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:147.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:139.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:132.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:142.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:137.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:130.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:124.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:113.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:126.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:120.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:115.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:120.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:138.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:137.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:130.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:124.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:113.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:121.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:119.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:109.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:105.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:117.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:115.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:133.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:128.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:122.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:117.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:112.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:120.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:117.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:112.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:122.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:111.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:121.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:111.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:117.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:132.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:143.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:137.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:130.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:124.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:135.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:124.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:148.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:147.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:139.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:142.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:137.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:130.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:123.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:112.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:126.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:123.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:140.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:135.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:143.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:138.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:119.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:109.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:105.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:115.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:90.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:87.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:85.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:83.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:90.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:109.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:121.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:132.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:148.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:158.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:147.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:139.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:149.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:159.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:155.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:163.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:178.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:178.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:157.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:180.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:157.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:159.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:159.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:168.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:181.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:169.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:159.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:180.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:181.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:162.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:181.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:178.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:180.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:162.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:168.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:178.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:181.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:178.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:187.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:154.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:170.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:160.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:178.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:156.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:147.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:139.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:132.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:136.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:132.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:119.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:109.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:105.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:101.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:96.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:93.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:90.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:105.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:101.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:108.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:96.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:113.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:108.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:100.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:97.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:94.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:91.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:87.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:96.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:112.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:109.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:120.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:125.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:134.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:147.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:164.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:153.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:144.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:152.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:160.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:155.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:157.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:148.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:140.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:133.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:126.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:120.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:115.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:114.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:111.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:96.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:93.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:90.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:88.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:86.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:84.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:82.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:97.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:94.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:91.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:107.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:96.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:93.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:121.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:124.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:118.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:113.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:108.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:100.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:97.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:94.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:91.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:86.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:90.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:88.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:85.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:84.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:82.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:80.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:79.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:103.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:115.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:105.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:98.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:87.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:85.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:100.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:97.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:94.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:91.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:89.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:87.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:85.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:100.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:97.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:113.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:110.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:106.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:102.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:99.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:92.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:90.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    
    FPS:87.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 50% 
    
    FPS:85.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 50% 
    
    FPS:83.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 71% 
    
    FPS:100.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 72% 
    0: 25% 
    
    FPS:115.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 62% 
    
    FPS:118.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 77% 
    0: 33% 
    
    FPS:113.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 68% 
    
    FPS:108.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 28% 
    
    FPS:104.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 29% 
    
    FPS:100.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 28% 
    
    FPS:119.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 88% 
    0: 64% 
    
    FPS:116.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 71% 
    0: 65% 
    
    FPS:111.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 77% 
    0: 47% 
    
    FPS:128.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 77% 
    0: 47% 
    
    FPS:124.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 78% 
    0: 47% 
    
    FPS:139.3 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 86% 
    0: 41% 
    
    FPS:150.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 78% 
    0: 56% 
    
    FPS:148.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 90% 
    0: 66% 
    
    FPS:154.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 90% 
    0: 66% 
    
    FPS:170.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 90% 
    0: 66% 
    
    FPS:183.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    0: 85% 
    
    FPS:192.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 88% 
    0: 68% 
    
    FPS:180.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 82% 
    0: 63% 
    
    FPS:181.8 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 83% 
    0: 64% 
    
    FPS:186.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 65% 
    0: 43% 
    
    FPS:189.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 86% 
    0: 80% 
    
    FPS:177.2 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    0: 83% 
    
    FPS:166.1 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 92% 
    
    FPS:166.9 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 93% 
    
    FPS:172.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 97% 
    
    FPS:176.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 98% 
    
    FPS:165.6 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 97% 
    
    FPS:172.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 96% 
    
    FPS:175.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 96% 
    
    FPS:180.4 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 94% 
    
    FPS:190.7 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 93% 
    
    FPS:179.5 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    0: 90% 
    
    FPS:183.0 	 AVG_FPS:96.2
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    0: 89% 
    
    FPS:182.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    0: 89% 
    
    FPS:184.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 87% 
    0: 83% 
    
    FPS:182.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 92% 
    0: 91% 
    
    FPS:194.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 95% 
    
    FPS:200.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 95% 
    
    FPS:206.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 95% 
    
    FPS:216.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 94% 
    
    FPS:205.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 94% 
    
    FPS:191.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    0: 91% 
    
    FPS:202.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    0: 84% 
    
    FPS:209.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    0: 84% 
    
    FPS:220.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    0: 92% 
    
    FPS:218.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    0: 90% 
    
    FPS:226.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:179.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:185.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:191.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.8 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:170.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:160.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:167.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:160.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.6 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:162.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:153.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:166.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:158.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:170.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:162.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:152.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:161.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:170.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.4 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:173.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:163.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.1 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:170.6 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:176.3 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:181.6 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.9 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:165.0 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.7 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:177.5 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.2 	 AVG_FPS:95.6
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.8 	 AVG_FPS:95.6
    Stream closed.
    
     cvWriteFrame 
    input video stream closed. 
     closing... closed!output_video_writer closed. 
    


```python
import cv2

path ='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/'
cap = cv2.VideoCapture(path+'testdemo1_result.avi')
 

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while (cap.isOpened):

    ret, frame = cap.read()

 

    if ret:

        cv2.imshow("video", frame)



        k = cv2.waitKey(33)

        if k == 27:

            break

    else:

        break

cap.release()

cv2.destroyAllWindows()

```

2) testdemo2


```python
! darknet detector demo custom/tl.data cfg/yolov4-tiny-custom.cfg  backup/yolov4-tiny-custom_best.weights -dont_show testdemo2.mp4 -i 0 -out_filename testdemo2_result.avi
```

     CUDNN_HALF=1 
    Demo
    net.optimized_memory = 0 
    mini_batch = 1, batch = 16, time_steps = 1, train = 0 
    nms_kind: greedynms (1), beta = 0.600000 
    nms_kind: greedynms (1), beta = 0.600000 
    
     seen 64, trained: 1070 K-images (16 Kilo-batches_64) 
    video file: testdemo2.mp4
    Video stream: 854 x 480 
    Objects:
    
    
    FPS:0.0 	 AVG_FPS:0.0
    Objects:
    
    
    FPS:7.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:28.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:47.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:59.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:60.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:76.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:111.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:105.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:101.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:116.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:134.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:145.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:152.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:144.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:158.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:171.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:174.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    

     CUDA-version: 10020 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     OpenCV version: 4.3.0
     0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070 SUPER 
       layer   filters  size/strd(dil)      input                output
       0 conv     32       3 x 3/ 2    416 x 416 x   3 ->  208 x 208 x  32 0.075 BF
       1 conv     64       3 x 3/ 2    208 x 208 x  32 ->  104 x 104 x  64 0.399 BF
       2 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
       3 route  2 		                       1/2 ->  104 x 104 x  32 
       4 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       5 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       6 route  5 4 	                           ->  104 x 104 x  64 
       7 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
       8 route  2 7 	                           ->  104 x 104 x 128 
       9 max                2x 2/ 2    104 x 104 x 128 ->   52 x  52 x 128 0.001 BF
      10 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      11 route  10 		                       1/2 ->   52 x  52 x  64 
      12 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      13 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      14 route  13 12 	                           ->   52 x  52 x 128 
      15 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      16 route  10 15 	                           ->   52 x  52 x 256 
      17 max                2x 2/ 2     52 x  52 x 256 ->   26 x  26 x 256 0.001 BF
      18 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      19 route  18 		                       1/2 ->   26 x  26 x 128 
      20 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      21 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      22 route  21 20 	                           ->   26 x  26 x 256 
      23 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      24 route  18 23 	                           ->   26 x  26 x 512 
      25 max                2x 2/ 2     26 x  26 x 512 ->   13 x  13 x 512 0.000 BF
      26 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      27 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
      28 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
      29 conv     42       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  42 0.007 BF
      30 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
      31 route  27 		                           ->   13 x  13 x 256 
      32 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
      33 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
      34 route  33 23 	                           ->   26 x  26 x 384 
      35 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
      36 conv     42       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  42 0.015 BF
      37 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    Total BFLOPS 6.800 
    avg_outputs = 300731 
     Allocate additional workspace_size = 26.22 MB 
    Loading weights from backup/yolov4-tiny-custom_best.weights...Done! Loaded 38 layers from weights-file 
    

    FPS:221.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:187.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:188.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:187.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:182.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:179.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:191.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:337.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:340.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:337.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:338.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:340.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:341.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:342.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 42% 
    
    FPS:277.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 38% 
    
    FPS:243.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 38% 
    
    FPS:245.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 41% 
    
    FPS:247.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 58% 
    0: 27% 
    
    FPS:264.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 26% 
    0: 27% , 6: 33% 
    
    FPS:262.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 26% 
    0: 27% , 6: 33% 
    
    FPS:262.7 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 26% 
    0: 26% , 6: 33% 
    
    FPS:257.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 45% 
    6: 69% 
    
    FPS:260.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 44% 
    6: 69% 
    
    FPS:260.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 39% 
    6: 61% 
    
    FPS:259.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 50% 
    6: 71% 
    
    FPS:264.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 51% , 6: 43% 
    6: 53% 
    
    FPS:272.6 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 45% 
    6: 37% 
    6: 35% 
    0: 47% , 6: 27% 
    
    FPS:269.2 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 45% 
    6: 37% 
    6: 35% 
    0: 46% , 6: 28% 
    
    FPS:268.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 46% 
    6: 36% 
    6: 34% 
    0: 46% 
    
    FPS:267.1 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 39% 
    
    FPS:272.8 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 46% 
    
    FPS:274.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 48% 
    
    FPS:268.5 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    6: 46% 
    
    FPS:265.4 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 67% 
    0: 65% 
    
    FPS:272.9 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 82% 
    
    FPS:280.3 	 AVG_FPS:182.5
    
     cvWriteFrame 
    Objects:
    
    0: 87% 
    
    FPS:278.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 87% 
    
    FPS:277.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 87% 
    
    FPS:273.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 90% 
    
    FPS:280.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 85% 
    6: 34% 
    
    FPS:287.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 35% 
    6: 56% 
    6: 43% 
    0: 81% , 6: 37% 
    
    FPS:284.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 35% 
    6: 57% 
    6: 43% 
    0: 81% , 6: 37% 
    
    FPS:277.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 62% , 6: 68% 
    6: 67% 
    6: 53% 
    
    FPS:267.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 76% 
    0: 69% , 6: 66% 
    6: 64% 
    
    FPS:255.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 72% 
    6: 70% 
    0: 91% 
    
    FPS:252.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 80% 
    6: 71% 
    0: 71% , 6: 59% 
    
    FPS:252.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 79% 
    6: 71% 
    0: 68% , 6: 63% 
    
    FPS:251.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 39% 
    6: 31% 
    0: 58% 
    
    FPS:251.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 69% 
    6: 35% 
    
    FPS:256.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 71% 
    
    FPS:255.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 54% 
    
    FPS:254.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 54% 
    
    FPS:250.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 73% 
    
    FPS:250.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 26% 
    0: 93% 
    0: 52% 
    
    FPS:256.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 37% , 6: 42% 
    6: 26% 
    0: 85% 
    
    FPS:254.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 63% 
    6: 50% 
    0: 84% 
    
    FPS:246.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 63% 
    6: 47% 
    0: 86% 
    
    FPS:248.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 30% , 6: 75% 
    6: 75% 
    6: 57% 
    
    FPS:250.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 93% 
    6: 57% 
    0: 69% , 6: 29% 
    
    FPS:256.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 82% 
    6: 93% 
    6: 40% 
    
    FPS:257.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    6: 95% 
    6: 59% 
    
    FPS:257.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    6: 95% 
    6: 59% 
    
    FPS:259.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 88% 
    6: 92% 
    
    FPS:257.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 78% 
    6: 54% 
    
    FPS:263.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 73% 
    6: 62% 
    
    FPS:265.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 85% 
    6: 37% 
    0: 91% 
    
    FPS:265.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 85% 
    6: 37% 
    0: 91% 
    
    FPS:264.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 84% 
    6: 32% 
    0: 93% 
    
    FPS:253.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 33% 
    6: 88% 
    6: 85% 
    
    FPS:263.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 76% 
    6: 27% 
    0: 97% 
    
    FPS:265.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 84% 
    6: 42% 
    0: 95% 
    
    FPS:264.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 84% 
    6: 41% 
    0: 95% 
    
    FPS:266.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 51% , 6: 83% 
    6: 80% 
    6: 39% 
    
    FPS:268.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 92% 
    0: 36% , 6: 88% 
    6: 79% 
    
    FPS:276.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 94% 
    6: 88% 
    6: 85% 
    
    FPS:275.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 81% 
    6: 59% 
    0: 88% , 6: 31% 
    
    FPS:273.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 81% 
    6: 59% 
    0: 88% , 6: 31% 
    
    FPS:271.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 54% 
    6: 69% 
    6: 65% 
    6: 38% 
    
    FPS:271.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    6: 87% 
    0: 40% , 6: 79% 
    
    FPS:278.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 89% 
    6: 89% 
    0: 28% , 6: 79% 
    
    FPS:271.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 89% 
    6: 90% 
    6: 81% 
    
    FPS:267.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 89% 
    6: 90% 
    6: 81% 
    
    FPS:266.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 79% 
    6: 91% 
    6: 91% 
    6: 62% 
    
    FPS:265.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 92% 
    6: 66% 
    0: 80% , 6: 43% 
    
    FPS:268.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 78% 
    0: 99% 
    0: 91% 
    
    FPS:267.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 54% , 6: 75% 
    0: 97% 
    0: 93% 
    
    FPS:263.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 55% , 6: 74% 
    0: 98% 
    0: 94% 
    
    FPS:261.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    0: 28% , 6: 64% 
    0: 90% 
    
    FPS:259.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    6: 75% 
    0: 88% 
    
    FPS:268.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 68% 
    6: 71% 
    0: 41% , 6: 54% 
    
    FPS:265.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 86% 
    0: 92% 
    0: 72% 
    
    FPS:258.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 87% 
    0: 93% 
    0: 72% 
    
    FPS:254.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    6: 97% 
    
    FPS:254.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 93% 
    0: 93% 
    0: 75% 
    
    FPS:263.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 92% 
    0: 77% 
    
    FPS:259.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 79% , 6: 33% 
    6: 96% 
    
    FPS:258.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 79% , 6: 33% 
    6: 96% 
    
    FPS:256.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 84% 
    6: 91% 
    
    FPS:249.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 82% 
    0: 36% , 6: 87% 
    
    FPS:251.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 52% , 6: 82% 
    6: 96% 
    
    FPS:242.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 73% , 6: 50% 
    6: 95% 
    
    FPS:241.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 72% , 6: 50% 
    6: 95% 
    
    FPS:242.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 54% 
    6: 90% 
    
    FPS:242.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 60% 
    6: 95% 
    
    FPS:241.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 81% 
    0: 68% , 6: 70% 
    
    FPS:242.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 79% , 6: 77% 
    0: 73% , 6: 73% 
    
    FPS:244.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 80% , 6: 75% 
    0: 75% , 6: 71% 
    
    FPS:248.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 81% 
    0: 42% , 6: 61% 
    
    FPS:251.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 69% , 6: 47% 
    0: 60% , 6: 38% 
    
    FPS:259.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 74% 
    6: 61% 
    
    FPS:257.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 81% 
    0: 47% , 6: 49% 
    
    FPS:256.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 81% 
    6: 49% 
    0: 47% 
    
    FPS:257.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 89% 
    0: 67% 
    
    FPS:259.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 36% 
    0: 80% 
    
    FPS:267.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 30% 
    0: 83% 
    
    FPS:265.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 90% 
    6: 43% 
    
    FPS:262.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 45% 
    0: 90% 
    
    FPS:263.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 46% 
    0: 83% 
    
    FPS:264.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 77% 
    
    FPS:272.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 87% 
    
    FPS:270.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 46% 
    
    FPS:269.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    0: 46% 
    
    FPS:271.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    6: 70% 
    
    FPS:272.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 51% 
    0: 95% 
    
    FPS:280.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 51% 
    0: 95% 
    
    FPS:277.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 51% 
    0: 95% 
    
    FPS:277.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 51% 
    0: 95% 
    
    FPS:284.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    6: 51% 
    0: 94% 
    
    FPS:279.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.4 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.9 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.8 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.5 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.0 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.6 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:337.3 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:339.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.1 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.7 	 AVG_FPS:278.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.6 	 AVG_FPS:278.8
    Stream closed.
    
     cvWriteFrame 
    input video stream closed. 
     closing... closed!output_video_writer closed. 
    


```python
import cv2

path ='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/'
cap = cv2.VideoCapture(path+'testdemo2_result.avi')
 

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while (cap.isOpened):

    ret, frame = cap.read()

 

    if ret:

        cv2.imshow("video", frame)



        k = cv2.waitKey(33)

        if k == 27:

            break

    else:

        break
        
cap.release()

cv2.destroyAllWindows()

```

3) testdemo3


```python
! darknet detector demo custom/tl.data cfg/yolov4-tiny-custom.cfg  backup/yolov4-tiny-custom_best.weights -dont_show testdemo3.mp4 -i 0 -out_filename testdemo3_result.avi
```

     CUDNN_HALF=1 
    Demo
    net.optimized_memory = 0 
    mini_batch = 1, batch = 16, time_steps = 1, train = 0 
    nms_kind: greedynms (1), beta = 0.600000 
    nms_kind: greedynms (1), beta = 0.600000 
    
     seen 64, trained: 1070 K-images (16 Kilo-batches_64) 
    video file: testdemo3.mp4
    Video stream: 854 x 480 
    Objects:
    
    
    FPS:0.0 	 AVG_FPS:0.0
    Objects:
    
    
    FPS:7.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:27.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:45.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:67.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:86.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:104.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:119.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:134.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:137.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:142.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:151.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:155.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:158.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:158.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:172.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.6 	 AVG_FPS:0.0
    

     CUDA-version: 10020 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     OpenCV version: 4.3.0
     0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070 SUPER 
       layer   filters  size/strd(dil)      input                output
       0 conv     32       3 x 3/ 2    416 x 416 x   3 ->  208 x 208 x  32 0.075 BF
       1 conv     64       3 x 3/ 2    208 x 208 x  32 ->  104 x 104 x  64 0.399 BF
       2 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
       3 route  2 		                       1/2 ->  104 x 104 x  32 
       4 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       5 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       6 route  5 4 	                           ->  104 x 104 x  64 
       7 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
       8 route  2 7 	                           ->  104 x 104 x 128 
       9 max                2x 2/ 2    104 x 104 x 128 ->   52 x  52 x 128 0.001 BF
      10 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      11 route  10 		                       1/2 ->   52 x  52 x  64 
      12 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      13 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      14 route  13 12 	                           ->   52 x  52 x 128 
      15 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      16 route  10 15 	                           ->   52 x  52 x 256 
      17 max                2x 2/ 2     52 x  52 x 256 ->   26 x  26 x 256 0.001 BF
      18 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      19 route  18 		                       1/2 ->   26 x  26 x 128 
      20 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      21 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      22 route  21 20 	                           ->   26 x  26 x 256 
      23 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      24 route  18 23 	                           ->   26 x  26 x 512 
      25 max                2x 2/ 2     26 x  26 x 512 ->   13 x  13 x 512 0.000 BF
      26 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      27 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
      28 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
      29 conv     42       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  42 0.007 BF
      30 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
      31 route  27 		                           ->   13 x  13 x 256 
      32 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
      33 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
      34 route  33 23 	                           ->   26 x  26 x 384 
      35 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
      36 conv     42       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  42 0.015 BF
      37 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    Total BFLOPS 6.800 
    avg_outputs = 300731 
     Allocate additional workspace_size = 26.22 MB 
    Loading weights from backup/yolov4-tiny-custom_best.weights...Done! Loaded 38 layers from weights-file 
    

    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 63% 
    
    FPS:249.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 39% 
    
    FPS:247.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 43% 
    
    FPS:249.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 74% 
    
    FPS:253.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 75% 
    
    FPS:255.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 81% 
    
    FPS:257.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 65% 
    
    FPS:259.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 67% 
    
    FPS:262.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 51% 
    
    FPS:260.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 56% 
    
    FPS:256.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 43% 
    
    FPS:244.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 27% 
    
    FPS:244.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 44% 
    
    FPS:245.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 46% 
    
    FPS:246.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 62% 
    
    FPS:246.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 45% 
    6: 51% 
    
    FPS:244.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 48% 
    
    FPS:242.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 36% 
    
    FPS:245.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 37% 
    
    FPS:246.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 51% 
    
    FPS:245.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 60% 
    
    FPS:255.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 75% 
    
    FPS:254.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 84% 
    
    FPS:256.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 35% 
    
    FPS:257.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 63% 
    
    FPS:258.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 76% 
    
    FPS:259.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 83% 
    
    FPS:260.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:259.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    6: 54% 
    
    FPS:260.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 27% , 6: 65% 
    
    FPS:257.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:255.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    
    FPS:256.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:258.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:256.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    
    FPS:257.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 81% 
    0: 72% 
    
    FPS:259.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:261.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    0: 90% 
    
    FPS:261.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 85% 
    0: 77% 
    
    FPS:257.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 61% 
    0: 45% 
    
    FPS:250.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 36% 
    
    FPS:243.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 70% 
    6: 33% 
    
    FPS:234.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 83% 
    0: 71% 
    
    FPS:236.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 48% , 6: 40% 
    
    FPS:241.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 49% , 6: 63% 
    
    FPS:244.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 48% 
    
    FPS:244.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 35% 
    
    FPS:244.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:187.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 66% 
    
    FPS:262.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 85% 
    
    FPS:263.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 58% 
    0: 65% , 6: 49% 
    
    FPS:265.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 25% 
    0: 54% , 6: 37% 
    
    FPS:264.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 42% 
    
    FPS:261.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 28% 
    
    FPS:262.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 64% 
    0: 41% 
    
    FPS:251.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 50% 
    0: 27% 
    
    FPS:244.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 78% 
    
    FPS:243.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 37% 
    
    FPS:240.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    0: 37% 
    
    FPS:240.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 65% 
    
    FPS:234.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 80% 
    
    FPS:233.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 86% 
    
    FPS:236.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 75% 
    
    FPS:239.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    
    FPS:244.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    
    FPS:246.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:243.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    
    FPS:238.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:235.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:238.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:241.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:245.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:249.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:248.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:253.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:259.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:264.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:264.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:267.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 83% 
    0: 97% 
    
    FPS:272.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 93% 
    
    FPS:272.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:265.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 100% 
    
    FPS:260.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:263.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:263.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:260.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:261.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:265.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 63% 
    
    FPS:266.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 82% 
    
    FPS:265.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    0: 47% 
    
    FPS:266.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:267.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:268.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 92% 
    
    FPS:264.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:264.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 86% 
    
    FPS:264.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 87% 
    
    FPS:263.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    
    FPS:259.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    
    FPS:258.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 77% 
    
    FPS:256.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 86% 
    
    FPS:255.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 32% 
    
    FPS:253.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 44% 
    
    FPS:236.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 43% 
    
    FPS:237.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 67% 
    
    FPS:236.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 52% 
    
    FPS:241.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 59% 
    
    FPS:241.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 40% 
    
    FPS:241.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 86% 
    
    FPS:241.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 51% 
    
    FPS:238.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 41% 
    
    FPS:238.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:239.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:240.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    
    FPS:241.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:240.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    
    FPS:240.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:242.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 83% 
    
    FPS:245.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 75% 
    
    FPS:247.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:249.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 90% 
    
    FPS:252.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    
    FPS:257.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 95% 
    
    FPS:257.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 80% 
    
    FPS:253.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 88% 
    
    FPS:246.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    
    FPS:240.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 96% 
    
    FPS:232.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:227.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:224.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:227.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:229.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:232.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:231.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:234.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:235.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:238.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:241.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:244.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    
    FPS:243.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 33% 
    
    FPS:246.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 100% 
    0: 37% 
    
    FPS:248.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 79% 
    0: 99% 
    0: 96% 
    
    FPS:248.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    0: 99% 
    0: 97% 
    
    FPS:248.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 73% 
    0: 59% 
    
    FPS:248.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 100% 
    0: 72% 
    
    FPS:249.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 91% 
    0: 57% 
    
    FPS:250.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 95% 
    
    FPS:248.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 57% 
    0: 97% 
    
    FPS:247.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 85% 
    
    FPS:240.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 99% 
    0: 74% 
    
    FPS:234.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 97% 
    
    FPS:226.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 93% 
    
    FPS:225.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 65% 
    0: 93% 
    
    FPS:230.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 85% 
    
    FPS:234.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 42% 
    0: 76% 
    
    FPS:233.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 71% 
    
    FPS:235.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 94% 
    
    FPS:237.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 91% 
    
    FPS:238.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 92% 
    0: 46% 
    
    FPS:238.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 98% 
    
    FPS:239.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    0: 36% 
    
    FPS:238.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.4 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.9 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.6 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.2 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.8 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.1 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.5 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:236.5
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.4 	 AVG_FPS:236.5
    Stream closed.
    
     cvWriteFrame 
    input video stream closed. 
     closing... closed!output_video_writer closed. 
    


```python
import cv2

path ='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/'
cap = cv2.VideoCapture(path+'testdemo3_result.avi')
 

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while (cap.isOpened):

    ret, frame = cap.read()

 

    if ret:

        cv2.imshow("video", frame)



        k = cv2.waitKey(33)

        if k == 27:

            break

    else:

        break

cap.release()

cv2.destroyAllWindows()

```


```python

```


```python
! darknet detector demo custom/tl.data cfg/yolov4-tiny-custom.cfg  backup/yolov4-tiny-custom_best.weights -dont_show testdemo4.mp4 -i 0 -out_filename testdemo4_result.avi
```

     CUDNN_HALF=1 
    Demo
    net.optimized_memory = 0 
    mini_batch = 1, batch = 16, time_steps = 1, train = 0 
    nms_kind: greedynms (1), beta = 0.600000 
    nms_kind: greedynms (1), beta = 0.600000 
    
     seen 64, trained: 1070 K-images (16 Kilo-batches_64) 
    video file: testdemo4.mp4
    Video stream: 854 x 480 
    Objects:
    
    
    FPS:0.0 	 AVG_FPS:0.0
    Objects:
    
    
    FPS:9.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:30.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:48.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:60.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:77.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:95.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:111.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:126.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:120.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:131.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:141.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:153.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:162.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:168.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:175.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:183.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 59% 
    
    FPS:199.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 45% 
    
    FPS:202.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 70% 
    
    FPS:215.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 85% 
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 89% 
    
    FPS:214.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 76% 
    
    FPS:211.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 75% 
    
    FPS:208.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 72% 
    
    FPS:202.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 49% 
    
    FPS:213.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:217.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 68% 
    
    FPS:221.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 68% 
    
    FPS:226.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    
    FPS:228.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    
    FPS:214.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 93% 
    
    FPS:209.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 66% , 2: 65% 
    
    FPS:202.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 66% , 2: 65% 
    
    FPS:202.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 98% 
    
    FPS:200.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 98% 
    
    FPS:207.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    
    FPS:205.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 98% 
    
    FPS:209.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 98% 
    
    FPS:215.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    
    FPS:220.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 97% 
    
    FPS:204.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 95% 
    
    FPS:201.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 81% 
    
    FPS:199.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 81% 
    
    FPS:197.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 92% 
    
    FPS:195.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 88% 
    
    FPS:201.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 87% 
    
    FPS:198.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.7 	 AVG_FPS:0.0
    
    

     CUDA-version: 10020 (11020), cuDNN: 7.6.5, CUDNN_HALF=1, GPU count: 1  
     OpenCV version: 4.3.0
     0 : compute_capability = 750, cudnn_half = 1, GPU: GeForce RTX 2070 SUPER 
       layer   filters  size/strd(dil)      input                output
       0 conv     32       3 x 3/ 2    416 x 416 x   3 ->  208 x 208 x  32 0.075 BF
       1 conv     64       3 x 3/ 2    208 x 208 x  32 ->  104 x 104 x  64 0.399 BF
       2 conv     64       3 x 3/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.797 BF
       3 route  2 		                       1/2 ->  104 x 104 x  32 
       4 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       5 conv     32       3 x 3/ 1    104 x 104 x  32 ->  104 x 104 x  32 0.199 BF
       6 route  5 4 	                           ->  104 x 104 x  64 
       7 conv     64       1 x 1/ 1    104 x 104 x  64 ->  104 x 104 x  64 0.089 BF
       8 route  2 7 	                           ->  104 x 104 x 128 
       9 max                2x 2/ 2    104 x 104 x 128 ->   52 x  52 x 128 0.001 BF
      10 conv    128       3 x 3/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.797 BF
      11 route  10 		                       1/2 ->   52 x  52 x  64 
      12 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      13 conv     64       3 x 3/ 1     52 x  52 x  64 ->   52 x  52 x  64 0.199 BF
      14 route  13 12 	                           ->   52 x  52 x 128 
      15 conv    128       1 x 1/ 1     52 x  52 x 128 ->   52 x  52 x 128 0.089 BF
      16 route  10 15 	                           ->   52 x  52 x 256 
      17 max                2x 2/ 2     52 x  52 x 256 ->   26 x  26 x 256 0.001 BF
      18 conv    256       3 x 3/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.797 BF
      19 route  18 		                       1/2 ->   26 x  26 x 128 
      20 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      21 conv    128       3 x 3/ 1     26 x  26 x 128 ->   26 x  26 x 128 0.199 BF
      22 route  21 20 	                           ->   26 x  26 x 256 
      23 conv    256       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x 256 0.089 BF
      24 route  18 23 	                           ->   26 x  26 x 512 
      25 max                2x 2/ 2     26 x  26 x 512 ->   13 x  13 x 512 0.000 BF
      26 conv    512       3 x 3/ 1     13 x  13 x 512 ->   13 x  13 x 512 0.797 BF
      27 conv    256       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x 256 0.044 BF
      28 conv    512       3 x 3/ 1     13 x  13 x 256 ->   13 x  13 x 512 0.399 BF
      29 conv     42       1 x 1/ 1     13 x  13 x 512 ->   13 x  13 x  42 0.007 BF
      30 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
      31 route  27 		                           ->   13 x  13 x 256 
      32 conv    128       1 x 1/ 1     13 x  13 x 256 ->   13 x  13 x 128 0.011 BF
      33 upsample                 2x    13 x  13 x 128 ->   26 x  26 x 128
      34 route  33 23 	                           ->   26 x  26 x 384 
      35 conv    256       3 x 3/ 1     26 x  26 x 384 ->   26 x  26 x 256 1.196 BF
      36 conv     42       1 x 1/ 1     26 x  26 x 256 ->   26 x  26 x  42 0.015 BF
      37 yolo
    [yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.05
    Total BFLOPS 6.800 
    avg_outputs = 300731 
     Allocate additional workspace_size = 26.22 MB 
    Loading weights from backup/yolov4-tiny-custom_best.weights...Done! Loaded 38 layers from weights-file 
    

     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:209.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 31% 
    
    FPS:227.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 31% 
    
    FPS:230.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 30% 
    
    FPS:231.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    0: 28% 
    
    FPS:215.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 78% 
    
    FPS:224.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 88% 
    
    FPS:223.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 66% , 2: 52% 
    
    FPS:224.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 72% , 2: 36% 
    
    FPS:225.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 72% , 2: 36% 
    
    FPS:228.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 86% , 2: 42% 
    
    FPS:226.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:214.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 32% 
    
    FPS:219.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:223.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.3 	 AVG_FPS:0.0
    
     cvWriteFrame 


```python
import cv2

path ='/Users/PC921/Traffic_lights/vcpkg/installed/x64-windows/tools/darknet/'
cap = cv2.VideoCapture(path+'testdemo4_result.avi')
 

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while (cap.isOpened):

    ret, frame = cap.read()

 

    if ret:

        cv2.imshow("video", frame)



        k = cv2.waitKey(33)

        if k == 27:

            break

    else:

        break

cap.release()

cv2.destroyAllWindows()
```

    
    Objects:
    
    
    FPS:218.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:186.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:191.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:191.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:193.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:184.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:202.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:191.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:187.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:195.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:189.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:197.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 25% 
    
    FPS:219.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 65% 
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 84% 
    
    FPS:227.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 84% 
    
    FPS:232.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 83% 
    
    FPS:238.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 92% 
    
    FPS:231.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 70% , 2: 40% 
    
    FPS:215.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 55% , 2: 47% 
    
    FPS:220.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 55% , 2: 47% 
    
    FPS:225.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 51% , 2: 39% 
    
    FPS:232.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 35% 
    
    FPS:233.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 49% 
    
    FPS:198.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 31% 
    
    FPS:203.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 25% 
    
    FPS:210.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 66% , 2: 45% 
    
    FPS:203.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 60% , 2: 88% 
    
    FPS:199.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 86% , 2: 69% 
    
    FPS:192.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 86% , 2: 68% 
    
    FPS:196.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 89% 
    
    FPS:204.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 93% 
    
    FPS:196.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 67% 
    
    FPS:195.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 73% , 2: 52% 
    
    FPS:194.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 73% , 2: 52% 
    
    FPS:193.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:192.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:198.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:190.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:204.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 35% , 2: 83% 
    
    FPS:214.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 89% 
    
    FPS:202.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 95% 
    
    FPS:198.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    
    FPS:196.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    
    FPS:193.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 97% 
    
    FPS:192.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 45% 
    1: 30% 
    1: 92% 
    
    FPS:188.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 45% 
    1: 30% 
    1: 92% 
    
    FPS:194.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 57% 
    1: 49% 
    
    FPS:193.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 57% 
    1: 49% 
    
    FPS:192.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 60% 
    
    FPS:199.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 52% , 2: 57% 
    
    FPS:188.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 87% , 2: 53% 
    
    FPS:195.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:194.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:201.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:206.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:232.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 80% 
    
    FPS:238.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 79% 
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 31% 
    
    FPS:214.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:210.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:215.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 80% 
    
    FPS:213.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 64% 
    
    FPS:218.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 64% 
    
    FPS:222.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    1: 55% 
    
    FPS:229.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:209.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:203.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:200.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:199.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:208.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:196.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:205.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:216.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:211.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:207.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 33% 
    
    FPS:206.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 37% 
    
    FPS:214.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 33% 
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 34% 
    
    FPS:225.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:218.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.5 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:222.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:223.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:227.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:230.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:226.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 84% 
    
    FPS:224.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 80% 
    
    FPS:224.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 71% 
    
    FPS:231.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 71% 
    
    FPS:234.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 71% 
    
    FPS:237.3 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 80% 
    
    FPS:220.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 87% 
    2: 31% 
    
    FPS:226.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 91% 
    2: 46% 
    
    FPS:231.7 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 91% 
    2: 46% 
    
    FPS:234.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 95% 
    2: 42% 
    
    FPS:239.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 95% 
    
    FPS:222.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 94% 
    
    FPS:219.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 95% 
    2: 46% 
    
    FPS:221.8 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    2: 95% 
    2: 45% 
    
    FPS:222.6 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:213.9 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.2 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.1 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.4 	 AVG_FPS:0.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:217.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:214.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:212.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:219.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:221.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:228.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:231.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:235.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 51% 
    1: 51% 
    
    FPS:252.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 40% , 7: 29% 
    
    FPS:256.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 45% 
    1: 42% 
    
    FPS:257.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 45% 
    1: 41% 
    
    FPS:255.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 83% 
    1: 25% 
    
    FPS:253.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 78% 
    1: 38% 
    
    FPS:261.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 91% 
    
    FPS:260.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 90% 
    
    FPS:256.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 90% 
    
    FPS:256.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 75% , 2: 29% 
    
    FPS:257.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 68% 
    
    FPS:266.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 88% 
    
    FPS:267.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 40% , 2: 51% 
    
    FPS:267.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 41% , 2: 51% 
    
    FPS:266.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 52% 
    
    FPS:275.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 43% 
    
    FPS:271.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 27% 
    
    FPS:279.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 61% , 2: 38% 
    
    FPS:282.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 44% , 2: 51% 
    
    FPS:276.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 72% 
    1: 25% , 2: 50% 
    
    FPS:277.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 73% 
    2: 51% 
    
    FPS:276.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 87% 
    
    FPS:274.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 86% 
    
    FPS:282.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 41% , 2: 35% 
    
    FPS:281.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 39% 
    
    FPS:277.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 34% 
    
    FPS:283.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:240.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 26% 
    
    FPS:270.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 26% 
    
    FPS:272.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 30% 
    
    FPS:271.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 40% 
    
    FPS:284.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 34% 
    
    FPS:283.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 89% 
    2: 58% 
    
    FPS:283.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 89% 
    2: 58% 
    
    FPS:279.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 80% 
    2: 75% 
    
    FPS:267.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 74% 
    1: 43% , 2: 27% 
    
    FPS:268.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 83% 
    1: 49% , 2: 52% 
    
    FPS:259.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 73% 
    2: 69% 
    
    FPS:252.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 73% 
    2: 69% 
    
    FPS:255.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 88% 
    1: 31% , 2: 87% 
    
    FPS:255.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 42% , 2: 89% 
    2: 83% 
    
    FPS:263.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 90% 
    1: 93% , 2: 34% 
    
    FPS:263.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 84% 
    1: 94% 
    
    FPS:264.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 84% 
    1: 94% 
    
    FPS:264.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 47% , 2: 42% 
    1: 87% 
    
    FPS:260.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 75% , 2: 46% 
    1: 90% 
    
    FPS:269.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 91% 
    1: 89% , 2: 31% 
    
    FPS:266.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 82% , 2: 33% 
    1: 87% 
    
    FPS:266.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 82% , 2: 33% 
    1: 87% 
    
    FPS:267.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 76% , 2: 42% 
    1: 82% 
    
    FPS:264.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 75% 
    1: 82% 
    
    FPS:273.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 93% 
    1: 87% , 2: 58% 
    
    FPS:273.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 82% , 2: 74% 
    
    FPS:270.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 83% , 2: 74% 
    
    FPS:270.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 97% 
    1: 95% 
    
    FPS:268.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 68% , 2: 81% 
    1: 96% 
    
    FPS:276.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 35% , 2: 91% 
    1: 97% 
    
    FPS:273.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 68% , 2: 44% 
    1: 96% 
    
    FPS:269.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 68% , 2: 44% 
    1: 96% 
    
    FPS:266.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 30% , 2: 90% 
    1: 84% , 2: 70% 
    
    FPS:262.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 98% 
    1: 86% , 2: 69% 
    
    FPS:270.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 74% , 2: 84% 
    
    FPS:267.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 86% , 2: 76% 
    
    FPS:265.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 85% , 2: 77% 
    
    FPS:261.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 89% , 2: 59% 
    
    FPS:251.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 97% 
    1: 99% 
    
    FPS:244.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 96% 
    
    FPS:237.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 96% 
    
    FPS:231.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 99% 
    1: 97% 
    
    FPS:227.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 56% , 2: 88% 
    1: 97% 
    
    FPS:228.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    2: 93% 
    1: 44% 
    
    FPS:232.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 54% , 2: 94% 
    1: 95% 
    
    FPS:236.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 54% , 2: 94% 
    1: 95% 
    
    FPS:238.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 58% , 2: 94% 
    1: 97% 
    
    FPS:240.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% , 2: 33% 
    1: 97% 
    
    FPS:251.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 91% , 2: 45% 
    1: 99% 
    
    FPS:252.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 93% 
    1: 99% 
    
    FPS:253.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 93% 
    1: 99% 
    
    FPS:255.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 69% 
    1: 50% 
    
    FPS:254.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    7: 50% 
    1: 69% 
    
    FPS:264.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 91% 
    1: 98% 
    
    FPS:264.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 92% , 2: 30% 
    1: 96% 
    
    FPS:262.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 92% , 2: 29% 
    1: 96% 
    
    FPS:264.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 97% 
    1: 99% 
    
    FPS:263.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 92% 
    1: 99% 
    
    FPS:273.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 96% , 2: 31% 
    1: 91% 
    
    FPS:274.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 88% 
    1: 91% 
    
    FPS:275.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 88% 
    1: 91% 
    
    FPS:275.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 93% 
    1: 95% 
    
    FPS:270.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 96% 
    
    FPS:270.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 90% , 2: 65% 
    1: 91% 
    
    FPS:263.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 54% 
    1: 76% 
    
    FPS:255.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 54% 
    1: 76% 
    
    FPS:248.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 72% 
    1: 53% 
    
    FPS:246.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 76% 
    1: 70% 
    
    FPS:254.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 87% 
    1: 71% 
    
    FPS:261.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 87% 
    1: 71% 
    
    FPS:265.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 66% 
    1: 78% 
    
    FPS:266.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 73% 
    1: 94% 
    
    FPS:269.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    1: 87% 
    
    FPS:262.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 79% , 2: 52% 
    
    FPS:257.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 79% , 2: 51% 
    
    FPS:252.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    1: 89% 
    
    FPS:254.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 90% 
    1: 97% 
    
    FPS:260.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 56% 
    0: 36% 
    1: 93% 
    2: 81% 
    
    FPS:258.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 30% 
    1: 35% , 2: 70% 
    1: 90% , 2: 33% 
    
    FPS:258.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 30% 
    1: 35% , 2: 70% 
    1: 90% , 2: 32% 
    
    FPS:251.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 26% , 2: 67% 
    1: 71% , 2: 52% 
    
    FPS:245.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 93% 
    2: 84% 
    0: 27% 
    
    FPS:254.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 30% , 2: 82% 
    1: 80% , 2: 61% 
    
    FPS:257.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 43% , 2: 61% 
    1: 95% 
    
    FPS:263.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 43% , 2: 61% 
    1: 95% 
    
    FPS:268.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 87% 
    2: 81% 
    
    FPS:270.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 92% 
    1: 94% 
    
    FPS:279.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    2: 74% 
    1: 84% 
    
    FPS:278.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 35% 
    
    FPS:307.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 38% 
    
    FPS:311.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 26% 
    
    FPS:301.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 32% 
    
    FPS:294.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:296.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 46% 
    
    FPS:295.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 45% 
    
    FPS:296.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 29% 
    1: 56% 
    
    FPS:268.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 29% 
    
    FPS:272.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 45% 
    
    FPS:273.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 45% 
    
    FPS:275.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 39% 
    
    FPS:279.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 36% 
    1: 66% 
    
    FPS:286.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 53% 
    
    FPS:287.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 64% 
    
    FPS:284.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 65% 
    
    FPS:286.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 60% 
    1: 82% 
    
    FPS:288.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 86% 
    1: 96% 
    
    FPS:294.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 87% 
    1: 93% 
    
    FPS:294.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 85% 
    1: 94% 
    
    FPS:287.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 85% 
    1: 94% 
    
    FPS:287.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 77% 
    1: 94% 
    
    FPS:286.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 64% 
    1: 77% 
    
    FPS:293.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    1: 87% 
    
    FPS:287.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 81% 
    1: 92% 
    
    FPS:293.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 96% 
    1: 94% 
    
    FPS:287.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 94% 
    
    FPS:275.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 94% 
    
    FPS:265.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 87% 
    1: 93% 
    
    FPS:258.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 93% 
    1: 83% 
    
    FPS:258.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 83% 
    1: 84% 
    
    FPS:254.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 56% 
    1: 92% 
    
    FPS:255.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 56% 
    1: 92% 
    
    FPS:256.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 91% 
    1: 44% 
    
    FPS:259.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 78% 
    
    FPS:268.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 75% 
    1: 97% 
    
    FPS:269.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 98% 
    
    FPS:264.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 99% 
    
    FPS:261.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 100% 
    1: 99% 
    
    FPS:264.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 57% 
    
    FPS:272.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 43% 
    
    FPS:273.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 31% 
    
    FPS:260.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 100% 
    1: 100% 
    
    FPS:269.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 100% 
    1: 100% 
    
    FPS:259.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 100% 
    1: 100% 
    
    FPS:251.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 100% 
    
    FPS:247.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 100% 
    1: 100% 
    
    FPS:257.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 98% 
    
    FPS:256.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 97% 
    1: 100% 
    
    FPS:253.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 97% 
    1: 100% 
    
    FPS:256.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 99% 
    
    FPS:258.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 100% 
    1: 100% 
    
    FPS:268.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 99% 
    
    FPS:269.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 99% 
    
    FPS:267.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 99% 
    1: 99% 
    
    FPS:267.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 98% 
    1: 99% 
    
    FPS:269.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 93% 
    1: 99% 
    
    FPS:278.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 94% 
    1: 94% 
    
    FPS:278.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 26% 
    1: 98% 
    
    FPS:275.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    1: 27% 
    1: 98% 
    
    FPS:275.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 75% 
    0: 63% 
    
    FPS:271.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 47% 
    0: 34% 
    0: 26% 
    
    FPS:280.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    0: 25% 
    
    FPS:275.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.6 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.3 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.8 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.9 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.4 	 AVG_FPS:182.8
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    0: 39% 
    
    FPS:290.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    0: 72% 
    
    FPS:286.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    0: 72% 
    
    FPS:284.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:239.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:238.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:236.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:233.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:225.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:220.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:224.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:229.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:234.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:237.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:251.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:243.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    1: 27% 
    
    FPS:325.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    1: 33% 
    
    FPS:311.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    1: 33% 
    
    FPS:303.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.8 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.2 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.4 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.6 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.3 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.5 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.0 	 AVG_FPS:274.0
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:338.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:337.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:337.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:339.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:340.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:339.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:336.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:335.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:264.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 29% 
    
    FPS:273.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:261.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:248.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:254.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:247.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:331.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:330.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:333.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:334.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:332.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:328.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:329.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 60% 
    
    FPS:288.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 60% 
    
    FPS:291.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 57% 
    
    FPS:292.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 35% 
    
    FPS:299.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 34% 
    
    FPS:310.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 30% 
    
    FPS:305.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 31% 
    
    FPS:303.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    0: 45% 
    
    FPS:302.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:252.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:249.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:244.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:241.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:242.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:245.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:255.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:291.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 33% 
    
    FPS:301.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 31% 
    
    FPS:301.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 26% 
    
    FPS:295.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 26% 
    
    FPS:297.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    0: 25% 
    
    FPS:303.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    0: 43% 
    
    FPS:305.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    0: 43% 
    
    FPS:306.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:312.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 28% 
    
    FPS:316.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 29% 
    
    FPS:318.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:322.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:325.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:327.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:319.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:320.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:321.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:323.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:326.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:324.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:317.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:316.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:318.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:306.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:272.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:299.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:315.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:290.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:296.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:298.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:300.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:297.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:301.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:302.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:308.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:314.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:310.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:313.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:311.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:303.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:309.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:307.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:305.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:304.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:295.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:275.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:279.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:277.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:259.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:257.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 34% 
    
    FPS:267.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:260.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:265.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:270.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:262.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:256.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 59% 
    
    FPS:251.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 28% 
    
    FPS:252.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:246.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 39% 
    
    FPS:251.1 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 39% 
    
    FPS:254.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:253.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 45% 
    
    FPS:261.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:266.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:269.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 39% 
    
    FPS:282.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 36% 
    
    FPS:289.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    1: 37% 
    
    FPS:294.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:284.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:276.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:271.2 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:268.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:274.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:278.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:282.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:288.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:289.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:287.8 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:280.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:285.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:292.7 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:293.6 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:294.4 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:286.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:281.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:283.0 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:273.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:267.9 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:263.3 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:258.5 	 AVG_FPS:278.3
    
     cvWriteFrame 
    Objects:
    
    
    FPS:250.9 	 AVG_FPS:278.3
    Stream closed.
    
     cvWriteFrame 
    input video stream closed. 
     closing... closed!output_video_writer closed. 
    
