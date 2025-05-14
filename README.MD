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
git clone https://www.modelscope.cn/l424102993/LLM_TQ_Tuned_model.git
## 替换其他路径
```

使用魔塔sdk下载
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('l424102993/LLM_TQ_Tuned_model', cache_dir = "./")
## 替换其他路径
```

## 使用方式

1.启动评分服务
```python
python evaluator_request.py
```

2.使用Evaluate_example.ipynb 调用api接口或本地模型进行单个模型单次的评分测试

## 批量评估

2.使用Evaluate_batch.ipynb 调用api接口或本地模型进行多个模型的评分测试
