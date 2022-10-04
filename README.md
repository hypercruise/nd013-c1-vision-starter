# Object Detection in an Urban Environment

This build has been tested with the following environment:

1) HW Specification
- CPU: AMD Ryzen 9 5950x 16-core
- GPU: Nvidia RTX 3090ti (Dual)
- RAM: 128GB
2) OS: Ubuntu 20.04.5 LTS
3) Nvidia Driver: 470.141.03
4) Cuda: 11.2
5) cuDNN: 8.1.0
6) Tensorflow: 2.6.0
7) Python: 3.9.13

### < Check Compatibility >
| Version | Python version | Compiler | Build tools | cuDNN | CUDA |
|---------|----------------|----------|-------------|-------|------|
|tensorflow-2.10.0 | 3.7-3.10 | GCC 9.3.1 | Bazel 5.1.1 | 8.1 | 11.2|
|tensorflow-2.9.0 | 3.7-3.10 | GCC 9.3.1 | Bazel 5.0.0 | 8.1 | 11.2|
|tensorflow-2.8.0 | 3.7-3.10 | GCC 7.3.1 | Bazel 4.2.1 | 8.1 | 11.2|
|tensorflow-2.7.0 | 3.7-3.9 | GCC 7.3.1 | Bazel 3.7.2 | 8.1 | 11.2|
|tensorflow-2.6.0 | 3.6-3.9 | GCC 7.3.1 | Bazel 3.7.2 | 8.1 | 11.2|
|tensorflow-2.5.0 | 3.6-3.9 | GCC 7.3.1 | Bazel 3.7.2 | 8.1 | 11.2|

* The official matrix is available on https://www.tensorflow.org/install/source

## Project Instructions

### Data for Local Machine
For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).
The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records.

### Data Structure
The data you will use for training, validation and testing is organized as follow:
```
../data/processed
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.

### Experiments Structure
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - experiment3/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `../data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

**Local Machine:**
The download_process.py is modified for using downloaded raw data due to download issue on local machine. Please the download the training data to the "data/raw" folder before running the script. 

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

* If you encounter a tensorflow error (AttributeError: module 'tensorflow.compat.v2.__internal__' has no attribute 'register_clear_session_function'), please downgrade 'keras' to 2.5.0rc with the following command:
```
pip install keras==2.5.0rc
```

## Instructions

### Exploratory Data Analysis

You should use the data in `../data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `../data/processed`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `../data/train/`, `../data/val/`, and `../data/test/`
* split the tf records files between these three folders by symbolically linking the files from `../data/waymo/` to `../data/train/`, `../data/val/`, and `../data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --source ../data/processed/ --destination ../data/
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `../nd013-c1-vision-starter/experiments/pretrained_model/`.
```
cd ../nd013-c1-vision-starter/experiments/pretrained_model/

wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir ../data/train/ --eval_dir ../data/val/ --batch_size 2 --checkpoint ../nd013-c1-vision-starter/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ../nd013-c1-vision-starter/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.
Open the new config file and change the third line to 'num_classes: 3'.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `../nd013-c1-vision-starter/experiments/reference` folder. Now launch the training process:
* a training process:
```
mv pipeline_new.config experiments/reference/

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path ../data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation_ref.gif
```

## Submission Template

### Project overview
This project is related to detect objects, e.g., vehicles, pedestrians, and cyclists, in an urban environment using Deep Learning.
In order to develop Self-Driving cars on efficient ways, it is essencial to use deep learning algorithms to detect objects and lane lines on the roads, and classify traffic light signals, etc.
Although we use the same dataset, the results will be different by selection of models and setting of configuration parameters.

Thus, I focused on learning the following items in this project:
1) How to setup Deep Learning development environments in a local machine. I believe one of the hardest parts of deep learning is setup the machine with GPUs, Cuda, CuDNN, Tensorflow and Python packages.
2) How to improve the performance of Deep Learning model by adjusting learning parameters.
3) How to select and debug a pre-trained model on the custom machine.


### Set up
Please refer the modified project instruction above.

### Dataset
#### Dataset analysis
The Waymo open dataset was used for this projects. More specifically 200 tfrecords were randomly selected.
The dataset contains various roads scenes weather conditions as follows:

![image](https://user-images.githubusercontent.com/113762711/193217461-35bff48b-f46d-4ea8-8738-eaa441e85537.png)
![image](https://user-images.githubusercontent.com/113762711/193217898-6b440d12-28d4-4bf7-a1eb-ac96f017d5cd.png)

#### Cross validation
The 200 tfrecords were splited for training, evaluation, and testing in a ratio of 80%:10%:10% using "create_splits.py" in this repository.


### Training
#### Reference experiment
The referecne experiment was conducted using the baseline model and configuration which were provided by Udacity.
```
Model: SSD ResNet50 V1 FPN 640x640 (RetinaNet50)
Num of Steps: 25k
Config: Not Changed
```

#### Results of Reference experiment 
The results of reference experiment was not good.
As the learning progressed, the loss reduction appeared very gradual, and as a result, it showed low performance even after the learning was completed.
![image](https://user-images.githubusercontent.com/113762711/193220750-45e2ecda-17d7-4ff4-bcf7-f246d806baec.png)


As shown the detecion image below, only two vehicles can be detected with very low confidence levels.
![image](https://user-images.githubusercontent.com/113762711/193221707-4abbf719-abcb-4130-9c88-5d1cef584215.png)


No vehicle was detected in the dark condition dataset.
![animation_ref](https://user-images.githubusercontent.com/113762711/193223187-d2fbc678-5089-4014-a1c3-1d4b8cb96e11.gif)


#### Exp1: Minor changes on batch size and data augmentation
The experiment 1 was conducted with the same model and slightly changed the configuration paramters as follows:
```
Model: SSD ResNet50 V1 FPN 640x640 (RetinaNet50)
Num of Steps: 25k
Changed config:
- batch_size: 2 --> 24 (Larger batch size for fast learning)
- data_augmentation_options {
    random_image_scale {
      min_scale_ratio: 0.3
      max_scale_ratio: 1.3
    }
  }
```
#### Results of Exp1 
The results of experiment 1 was highly improved.
The loss reduction show relatively stiff, and it showed higher performance than the referecne

![image](https://user-images.githubusercontent.com/113762711/193225051-41d168bd-020c-4c06-9f1a-6a4671fa0580.png)


As shown the detecion image below, much more vehicles can be detected with higher confidence levels.
But, some vehicles were still not detected.
![image](https://user-images.githubusercontent.com/113762711/193225698-a4f819b4-132a-475d-a0b2-38d96eac88d6.png)


Now the vehicles can be detected in the dark condition dataset.
![animation_exp1](https://user-images.githubusercontent.com/113762711/193226293-912b4e31-a7d9-426a-bfd5-7e7813eac61f.gif)


#### Exp2: Changes on learning parameters with the same model
The experiment 2 was conducted with the same model and the configuration paramters as follows:
```
Model: SSD ResNet50 V1 FPN 640x640 (RetinaNet50)
Num of Steps: 300k
Changed config:
- batch_size: 24 --> 16 (Sliightly lower batch size to reduce overfitting)
- data_augmentation_options
  --> remove random_crop_image
- learning rate parameters changed:
  learning_rate_base: 0.04 --> 0.08
  total_steps: 25000 --> 300000
  warmup_learning_rate: 0.013333 --> 0.001
  warmup_steps: 2000 --> 2500
```
#### Results of Exp2
The results of experiment 2 show no major improvement than Experiment 1.
The total loss showed much lower than the experiment 2, but the performance enhancement was limited due to overfitting. 

![image](https://user-images.githubusercontent.com/113762711/193229554-11e4eba5-71d6-454a-a030-00315e0bf71a.png)

As shown the detecion image below, much more vehicles can be detected with very high confidence levels.
But, most vehicles were not detected. Even worse than experiment 1.

![image](https://user-images.githubusercontent.com/113762711/193229879-5d9954bd-2168-4f3d-9be1-4806b7d01768.png)


This model can detect the vehicles with very high confidence in the dark condition dataset.
![animation_exp2](https://user-images.githubusercontent.com/113762711/193230793-2f038738-5463-4919-93c2-1518a1eb31aa.gif)


#### Exp3: With new pre-trained model
I believe one of the most important feature of deep learning models in Self-Driving car is real-time performance.
Thus, I took a different pre-trained model, so called, CenterNet for the last experiment.
Because it show the fasted performance among the pre-trained models in the model zoo (see the table capture below).

![image](https://user-images.githubusercontent.com/113762711/193818028-991f1269-78db-4a97-b74b-5cc6f21c8f82.png)
![image](https://user-images.githubusercontent.com/113762711/193817836-b66e4e58-c36b-4f98-afc9-37bc03415634.png)
![image](https://user-images.githubusercontent.com/113762711/193818196-0834d97d-02a9-4120-a63c-33a72f26467c.png)


The configuration paramters were set as follows:

```
Model: CenterNet MobileNetV2 FPN 512x512
Num of Steps: 500k
Changed config:
- batch_size: 16

Changed parameters for debugging:
- use_separable_conv: true (Added)
- fine_tune_checkpoint_version: V2 (Added)
- random_horizontal_flip (Removed)

Augmentaiton options
- random_crop_image
- random_adjust_hue
- random_adjust_contrast
- random_adjust_saturation
- random_adjust_brightness
- random_absolute_pad_image
```

#### Examples of augmention effects

- random_absolute_pad_image is used to simulate an obscured image occured by signs, street trees, etc. on urban roads.

![image](https://user-images.githubusercontent.com/113762711/193819724-c8f8613b-2fcd-4d1b-b771-8a7898aefbae.png)

- random_crop_image is used to simulate offet views when turning an ego vehicle the street corners can be located on the vehicle center. The random_crop can generate these offet view points.

![image](https://user-images.githubusercontent.com/113762711/193821538-7163c491-4618-4534-b9a6-a9cf7c335e0e.png)

- The image below is combined scene of pad_image and crop_image.

![image](https://user-images.githubusercontent.com/113762711/193819943-1ab2be3c-f6ba-4c68-a1c6-99082e4c8ee5.png)

- The color related augmentation such as random_adjust_hue, random_adjust_contrast, random_adjust_saturation, and random_adjust_brightness, can be used to simulate different brightness conditions such as night, sun set, sun reflection, etc.

![image](https://user-images.githubusercontent.com/113762711/193821899-20dca496-0003-4f78-aea5-c868d3b87913.png)


#### Results of Exp2
The results of experiment 3 show no major improvement than Experiment 1.
The total loss showed much lower than the experiment 2, but the performance enhancement was limited due to overfitting. 

![image](https://user-images.githubusercontent.com/113762711/193233634-6d3b0b1a-48b8-4dbf-9109-2094f82031a4.png)


As shown the detecion image below, this model can detect most vehicles, but the confidence levels were relatively low. 

![image](https://user-images.githubusercontent.com/113762711/193233821-da251e37-b876-44cc-a6f4-f5c6909724e0.png)


This model can detect the vehicles in the dark condition dataset, but the confidence levels were relatively low.
![animation_exp3](https://user-images.githubusercontent.com/113762711/193233905-8397686a-b963-4254-815a-a5942ba00a5e.gif)
