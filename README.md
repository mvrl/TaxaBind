# TaxaBind: A Unified Embedding Space for Ecological Applications
<div align="center">
<img src="imgs/taxabind_logo.png" width="250">

[![arXiv](https://img.shields.io/badge/arXiv-2411.00683-red)](https://arxiv.org/pdf/2411.00683)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://vishu26.github.io/taxabind/index.html)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow
)](https://huggingface.co/collections/MVRL/taxabind-664be83f97dd6a0c4a5d2293)
[![Hugging Face Models](https://img.shields.io/badge/TaxaBind-Datasets-blue
)](https://huggingface.co/collections/MVRL/taxabind-datasets-672292b5a68683b4272950db)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Spaces-yellow?style=flat&logo=hug)](https://huggingface.co/spaces/MVRL/taxabind-demo)</center>

[Srikumar Sastry*](https://vishu26.github.io/),
[Subash Khanal](https://subash-khanal.github.io/),
[Aayush Dhakal](https://scholar.google.com/citations?user=KawjT_8AAAAJ&hl=en),
[Adeel Ahmad](https://adealgis.wixsite.com/adeel-ahmad-geog),
[Nathan Jacobs](https://jacobsn.github.io/)
(*Corresponding Author)

WACV 2025
</div>

This repository is the official implementation of [TaxaBind](https://arxiv.org/pdf/2411.00683).
TaxaBind is a suite of multimodal models useful for downstream ecological tasks covering six modalities: ground-level image, geographic location, satellite image, text, audio, and environmental features.

![](imgs/framework_2.jpg)


## 🎯 Zero-Shot Image Classification
![](imgs/results.png)
Our framework outperforms the state-of-the-art in both unimodal (BioCLIP, ArborCLIP) and multimodal setting (ImageBind).

## 🔥 Large Mulitmodal Ecological Datasets

* We release [TaxaBench-8k](https://huggingface.co/datasets/MVRL/TaxaBench-8k), a truly multimodal dataset containing six paired modalities for evaluating large ecological models.
* We release [iSatNat](https://huggingface.co/datasets/MVRL/iSatNat), containing 2.7M pairs of satellite images and ground-level species images.
* We release [iSoundNat](https://huggingface.co/datasets/MVRL/iSoundNat), containing 88,130 pairs of audio and ground-level species images.

## ⚙️ Usage
Our pretrained models are made available through `rshf` and `transformers` package for easy inference.

Load and initialize taxabind config:
```python
from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind

config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
taxabind = TaxaBind(config)
```

📎 Loading ground-level image and text encoders:
```python
# Loads open_clip style model

model = taxabind.get_image_text_encoder()
tokenizer = taxabind.get_tokenizer()
processor = taxabind.get_image_processor()
```

🛰️ Loading satellite image encoder:
```python
sat_encoder = taxabind.get_sat_encoder()
sat_processor = taxabind.get_sat_processor()
```

📍 Loading location encoder:
```python
location_encoder = taxabind.get_location_encoder()
```

🔈 Loading audio encoder:
```python
audio_encoder = taxabind.get_audio_encoder()
audio_processor = taxabind.get_audio_processor()
```

🌦️ Loading environmental encoder:
```python
env_encoder = taxabind.get_env_encoder()
env_processor = taxabind.get_env_processor()
```

📑 Citation

```bibtex
@inproceedings{sastry2025taxabind,
    title={TaxaBind: A Unified Embedding Space for Ecological Applications},
    author={Sastry, Srikumar and Khanal, Subash and Dhakal, Aayush and Ahmad, Adeel and Jacobs, Nathan},
    booktitle={Winter Conference on Applications of Computer Vision},
    year={2025},
    organization={IEEE/CVF}
}
```


## 🔍 Additional Links
Check out our lab website for other interesting works on geospatial understanding and mapping:
* Multi-Modal Vision Research Lab (MVRL) - [Link](https://mvrl.cse.wustl.edu/)
* Related Works from MVRL - [Link](https://mvrl.cse.wustl.edu/publications/)
