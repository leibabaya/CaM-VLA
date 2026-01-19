# CaM-VLA
Class-aware Morphological Vision–Language Alignment for Cervical Cell Image Classification.
## Abstract
Cervical cytology screening is crucial for the early detection of cervical cancer, and deep learning–based cell image classification is an important approach to automated screening. However, end-to-end prediction lacks interpretability and tend to exploit staining variations and statistical image features, thus is not consistent with clinical diagnostic logic rested upon morphological criteria such as the nuclear-to-cytoplasmic ratio and nuclear membrane morphology. To address this issue, we propose a morphology semantics-guided framework for cervical cell image classification. We map discrete class labels to class-level morphological text descriptions according to the Bethesda System (TBS), and adopt an N-to-1 correspondence paradigm between image instances and class description. Based on this, we design a Class-aware Morphological Vision–Language Alignment (CaM-VLA) module that leverages bidirectional multi-granular cross-modal attention (Text-to-Vision and Vision-to-Text) and class-aware contrastive learning to establish fine-grained correspondences between morphological semantics and image features. We use cross-modal similarity to reform the classification to cross-modal matching, which is consistent with the working principle of CaM-VLA. In-domain and cross-domain experiments show that explicitly incorporating morphology semantics improves classification performance. Visualization results show that the model can correctly associate morphological concepts with local image features, validating its interpretability. This work provides a new perspective to interpretable and transferable cervical cell image recognition.
### Dataset downloading
Datasets we used are as follows:
- **[HiCervix](https://zenodo.org/records/11081816)**
- **[CRIC](https://database.cric.com.br)**
### Data Preprocessing
We preprocessed these datasets and split the dataset into train/val/test set using the code in `preprocess`.
### Training
We trained CaM-VLA on HiCervix using this command:
```
cd mgca/models/CAM_VLA
CUDA_VISIBLE_DEVICES=0,1 python mgca_module_semantic_vf_structured.py --gpus 2 --strategy ddp
```
### Requirements
### Requirements
* Python 3.9
* PyTorch 2.6+
* **pytorch-lightning==1.9.5** 
* **transformers==4.18.0** 
### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
