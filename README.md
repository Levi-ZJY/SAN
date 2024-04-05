
# SAN: Structure-Aware Network for Complex and Long-tailed Chinese Text Recognition

The official code of SAN (ICDAR 2023)


## Runtime Environment
- Using the dependencies
    ```
    pip install -r requirements.txt
    ```




## Datasets

Test datasets can be downloaded from [Datasets BaiduNetDisk (passwd: 7zmh)](https://pan.baidu.com/s/1QVIGzY9OxKZZpOMbCCBKoA).

Move these files to : ```SAN/data/```

## Model Result

Get the model training result from [Model Result BaiduNetDisk (passwd: 69th)](https://pan.baidu.com/s/1P6tUyv-qEbc10cR7oUJ5eA).

Move these files to : ```SAN/workdir/```



## Training

1. Pre-train vision model
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/pretrain_vision_model.yaml
    ```
2. Pre-train language model
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/pretrain_language_model.yaml
    ```
3. Train ABINet
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config=configs/train_abinet.yaml
    ```


## Notice

The code I provided can be directly trained or tested on the ```Web dataset```. If you would like to train or test it on the ```Scene dataset```, you need to modify the 'max_radical_length' parameter in the code. 

Specifically, please make changes in the following two parts of the code:

- SAN/configs/template.yaml <u>line 22</u>:
  
  ```max_length_radical: 33 -> max_length_radical: 39```

- SAN/dataset.py <u>line 221</u>:
  
  ```max_length:int=33 -> max_length:int=39 ```

- SAN/losses.py <u>line 24</u>:
  
  ```self.max_length_radical = 33 -> self.max_length_radical = 39 ```



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




## Citation
If you find our method useful for your reserach, please cite
```bash 
@inproceedings{Zhang2023SANSN,
  title={SAN: Structure-Aware Network for Complex and Long-Tailed Chinese Text Recognition},
  author={Junyi Zhang and Chang Liu and Chun Yang},
  booktitle={IEEE International Conference on Document Analysis and Recognition},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:261102024}
}
 ```