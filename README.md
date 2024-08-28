# Fish Classification 
## Usage

For instance, to run data augmentation on **Fish_Dataset**: 
```
python main.py --train_dir ./Fish_Dataset/Dataset --fractal_dir ./deviantart --prompts "trout,shrimp"
```

For instance, to run data augmentation on **NA_Fish_Dataset**: 
```
python main.py --train_dir ./NA_Fish_Dataset --fractal_dir ./deviantart --prompts "Sea-Bass,Striped-Red-Mullet"
```

To **fine-tune** Vision Transformer:
```
python ViT.py
```

To check for performance:
```
python Inference.py
```

## Cited
#### A Large Scale Fish Dataset
> @inproceedings{ulucan2020large,
> title={A Large-Scale Dataset for Fish Segmentation and Classification},
> author={Ulucan, Oguzhan and Karakaya, Diclehan and Turkan, Mehmet},
> booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
> pages={1--5},
> year={2020},
> organization={IEEE}
> }

> * O.Ulucan , D.Karakaya and M.Turkan.(2020) A large-scale dataset for fish segmentation and classification.
> In Conf. Innovations Intell. Syst. Appli. (ASYU)

#### DiffuseMix : Label-Preserving Data Augmentation with Diffusion Models (CVPR'2024)
> @article{diffuseMix2024,
> title={DIFFUSEMIX: Label-Preserving Data Augmentation with Diffusion Models},
> author={Khawar Islam, Muhammad Zaigham Zaheer, Arif Mahmood, Karthik Nandakumar},
> booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
> year={2024}
> }

#### Fish-Pak Species Dataset from Pakistan for Visual Features Based Classification
> Shah, Syed Zakir Hussain ; Rauf, Hafiz Tayyab; IkramUllah, Muhammad ; Bukhari, Syed Ahmad Chan ;  Khalid, Malik Shahzaib; Farooq, 
> Muhammad ; Fatima, Mahroze  (2019), “Fish-Pak: Fish Species Dataset from Pakistan for Visual Features Based Classification”, Mendeley 
> Data, V3, doi: 10.17632/n3ydw29sbz.3

#### Images used to train Amazonian fish classification model
> Dikow, Rebecca (2023). Images used to train Amazonian fish classification model. Office of the Chief Information Officer. Dataset. 
> https://doi.org/10.25573/data.17314730.v1

#### Car Images Dataset
> [Here](<https://www.kaggle.com/datasets/kshitij192/cars-image-dataset>) is the dataset from kaggle. 

#### Fractal Dataset
> This is the [Fractal Dataset](<https://drive.google.com/drive/folders/1uxK7JaO1NaJxaAGViQa1bZfX6ZzNMzx2>) to download. 
> This is provided by the "DiffuseMix : Label-Preserving Data Augmentation with Diffusion Models" paper.