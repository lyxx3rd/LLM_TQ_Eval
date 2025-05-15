## 配置环境

python = 3.12.9

安装环境
```bash
pip install -r requirements.txt
```
## 模型下载

Qwen_7B_Review_Tuned_model : https://www.modelscope.cn/l424102993/LLM_TQ_Tuned_model.git

Regression_model_base : https://www.modelscope.cn/iic/nlp_bert_backbone_base_std

Regression_model_regression : https://www.modelscope.cn/l424102993/LLM_TQ_Regression_model.git

使用git下载
```bash
cd data/models
git clone https://www.modelscope.cn/l424102993/LLM_TQ_Tuned_model.git
git clone https://www.modelscope.cn/iic/nlp_bert_backbone_base_std
git clone https://www.modelscope.cn/l424102993/LLM_TQ_Regression_model.git
## 替换其他路径
```

使用魔塔sdk下载
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('l424102993/LLM_TQ_Tuned_model', cache_dir = "./data/models/")
model_dir = snapshot_download('iic/nlp_bert_backbone_base_std', cache_dir = "./data/models/")
model_dir = snapshot_download('l424102993/LLM_TQ_Regression_model.gitl', cache_dir = "./data/models")
## 替换其他路径
```
## 使用方式

1. 修改config.ymal

修改模型目录以及cuda

2.启动评分requests服务
```python
python evaluator_request.py
```

3.使用Evaluate_example.ipynb 调用api接口或本地模型进行单个模型单次的评分测试

## 批量评估

1.使用Evaluate_batch.ipynb 调用api接口或本地模型进行多个模型的评分测试
