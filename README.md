# Radiology Report Generation with a Learned Knowledge Base and Multi-modal Alignment

基于自学习知识库和多模态对其机制的医学报告生成

## Requirements
- `Python >= 3.6`
- `Pytorch >= 1.7`
- `torchvison`
- [Microsoft COCO Caption Evaluation Tools](https://github.com/tylin/coco-caption)
- [CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler)

`conda activate tencent`

## Data

Download IU and MIMIC-CXR datasets, and place them in `data` folder.

- IU dataset from [here](https://iuhealth.org/find-medical-services/x-rays)
- MIMIC-CXR dataset from [here](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
    
    
## Folder Structure
- config : setup training arguments and data path
- data : store IU and MIMIC dataset
- models: basic model and all our models
- modules: 
    - the layer define of our model 
    - dataloader
    - loss function
    - metrics
    - tokenizer
    - some utils
- pycocoevalcap: Microsoft COCO Caption Evaluation Tools

## Training and Testing
- The validation and testing will run after training.
- More options can be found in `config/opts.py` file.
- The model will be trained using command：
    - $dataset_name:
        - iu: IU dataset
        - mimic: MIMIC dataset
    1. full model
    
        ```
        python main.py --cfg config/{$dataset_name}_resnet.yml --expe_name {$experiment name} --label_loss --rank_loss --version 12
        ```
        
    2. basic model
    
        ```
        python main_basic.py --cfg config/{$dataset_name}_resnet.yml --expe_name {$experiment name} --label_loss --rank_loss --version 91
        ```
        
    3. our model without the learned knowledge base
    
        ```
        python main.py --cfg config/{$dataset_name}_resnet.yml --expe_name {$experiment name} --label_loss --rank_loss --version 92
        ```
        
    4. for the model without multi-modal alignment
        You remove `--label_loss` or `--rank_loss` from the commonds.

## Citation
Shuxin Yang, Xian Wu, Shen Ge, ZhuoZhao Zheng, S. Kevin Zhou, Li Xiao,Radiology Report Generation with a Learned Knowledge Base and Multi-modal Alignment. Medical Image Analysis,2023

## Contact
If you have any problem with the code, please contact Shuxin Yang(aspenstarss@gmail.com) or Li Xiao(andrew.lxiao@gmail.com).
