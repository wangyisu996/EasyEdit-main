###📂 Data Preparation###
The datasets used can be found in Google Drive Link (ZsRE, Hallucination, Temporal)

Each dataset contains both an edit set and a train set.


Setup

This codebase uses Python 3.9, you can create a conda env:
conda create -n lifelong_edit python=3.9

conda activate lifelong_edit

pip install -r requirements.txt 

### 📂 Data Preparation###

The datasets used can be found in [Google Drive Link](https://drive.google.com/file/d/1YtQvv4WvTa4rJyDYQR2J-uK8rnrt0kTA/view?usp=sharing) (ZsRE, Hallucination, Temporal)

**Config**

- For reproducing experimental results, please refer to [config.yaml](https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/llama-7b.yaml), which contains the configuration parameters used for the run. Each parameter is explained in the configuration file, and we recommend using the default parameters. If you need to reproduce WISE-Retrieve, set `retrieve=True.
- We now provide preliminary support for chat templates. You can enable this feature by adding `use_chat_template: True` in the configuration and we provide an example on llama-3-8b(https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/llama-3-8b.yaml#L31). 
- We now support using WISE for knowledge editing on some of the latest models such as `LlaMa 3.1` and `Qwen2.5`, if you want to edit on `Qwen2` just apply the [Qwen2.5-7b.yaml](https://github.com/zjunlp/EasyEdit/blob/main/hparams/WISE/qwen2.5-7b.yaml).
注意：关于模型的超参调节在main/hparams/WISE/llama-3-8b.yaml（我把所有的超参的调节都集中在这里了）

### 📂Running experiments###
运行main/examples/run_wise_editing.py （k=case的数量，直接在这里改，还有数据集地址）
easyeditor/models/wise/WISE.py修剪side memory操作，蒸馏loss，改变mask方式都在这个文件里改，搜索wys可看到所有我添加的部分
