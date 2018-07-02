# 2DFAN4

an implement of 2DFAN4 proposed in the paper ["How far are we from solving the 2D & 3D face alignment problem?"](https://arxiv.org/abs/1703.07332) facial landmarker algorithm with caffe2 (pytorch) in C++

### introduction

the project implements two models proposed in the paper ["How far are we from solving the 2D & 3D face alignment problem?"](https://arxiv.org/abs/1703.07332). one model is 2DFAN4 which locates 2D landmarks from RGB facial image. another model is ResNetDepth which predicts the depth in camera coordinate system from given 2D landmarks in image coordinate system and the original RGB facial image.

### how to build

you need [caffe2 model helper library](https://github.com/breadbread1984/caffe2_cpp_tutorial) to build the project. install the library before compile this one.

build everything with the following command

```bash
make
```

### how to play

a pretrained 2DFAN4 model can be downloaded from Baidu Cloud at

> 链接:https://pan.baidu.com/s/12PRtDjIvKigmX3NeoDmTww 密码:w44a

placing the two files in 2DFAN4_params, you can play the facial landmarker directly.

you can test the 2D landmarker on single image with

```bash
./2DLandmarker -i <path/to/image>
```

you can also try to test the landmarker on you webcam with

```bash
./2DLandmarker
```

### how to train

if you want to train the model youself, you need to follow the instructions below.

#### prepare training data

2DFAN4 is trained on 300W-LP which can be downloaded from [here](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm). the label files of the original dataset is in matlab format. you need the following matlab code to convert it into csv format which can be read by C++ easily.

```matlab
function  convert()
convertInDir('AFW');
convertInDir('HELEN');
convertInDir('IBUG');
convertInDir('LFPW');
end

function  convertInDir(d)
cd(d);
Files=dir('*.mat');
for  k=1:length(Files)
	data  =  load(Files(k).name);
	[pathstr,name,ext]=fileparts(Files(k).name);
	outfilename=strcat(name,'.csv');
	csvwrite(outfilename,data.pts_3d);
end
cd  '..'; 
end
```

move the images to the corresponding directory containing label files and start training with

```bash
./train_2DFAN4 -i <path/to/dataset>
```

ResNetDepth is trained on 300W-3D which can be downloaded from the same location of 300W-LP. You need to convert the matlab version label to csv in the similar way. then move images to the corresponding directory containing label files and start training with

```bash
./train_ResNetDepth -i <path/to/dataset>
```

