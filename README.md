
# SAN-Code

## Runtime Environment
- Using the dependencies
    ```
    pip install -r requirements.txt
    ```




## Test Datasets

Test datasets can be downloaded from [Web_test and Scene_test WeiYun (passwd: funf3i)](https://share.weiyun.com/MnIrRgo4).

Move these files to : ```SAN/data/```

## Model Result

Get the model training result from [Model Result WeiYun (passwd: zmcvf4)](https://share.weiyun.com/k7DUruUp).

Move these files to : ```SAN/workdir/```

## Evaluation

Web datatet:

Radical length : 33

- ABINet-TreeSim(SAN):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet.yaml --phase test --checkpoint=workdir/SAN-web-final/best-train-abinet.pth --test_root=data/web/web_val/ --model_eval=alignment --image_only
```

- VM-TreeSim:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/pretrain_vision_model.yaml --phase test --checkpoint=workdir/VisionTreeSim-web-final/best-pretrain-vision-model.pth --test_root=data/web/web_val/ --model_eval=vision --image_only
```

&nbsp;

Scene dataset:

Radical length : 39

- ABINet-TreeSim(SAN):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet.yaml --phase test --checkpoint=workdir/SAN-scene-final/best-train-abinet.pth --test_root=data/scene/scene_val/ --model_eval=alignment --image_only
```

- VM-TreeSim:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/pretrain_vision_model.yaml --phase test --checkpoint=workdir/VisionTreeSim-scene-final/best-pretrain-vision-model.pth --test_root=data/scene/scene_val/ --model_eval=vision --image_only
```

## Notice

The code I provided can be directly tested on the ```Web dataset```. If you would like to test it on the ```Scene dataset```, you need to modify the 'max_radical_length' parameter in the code. 

Specifically, please make changes in the following two parts of the code:

- ./configs/template.yaml <u>line 22</u>:
  
  ```max_length_radical: 33 -> max_length_radical: 39```

- ./dataset.py <u>line 221</u>:
  
  ```max_length:int=33 -> max_length:int=39 ```