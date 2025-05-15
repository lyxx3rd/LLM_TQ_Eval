## 配置环境

python = 3.12.9;

磁盘占用约: 17G;

显存占用约:18G;

安装环境
```bash
pip install -r requirements.txt
```
## 模型下载

Qwen_7B_Review_Tuned_model : https://www.modelscope.cn/l424102993/LLM_TQ_Tuned_model.git

Regression_model_base : https://www.modelscope.cn/iic/nlp_bert_backbone_base_std.git

Regression_model_regression : https://www.modelscope.cn/l424102993/LLM_TQ_Regression_model.git

使用git下载(与魔塔sdk二选一)
```bash
cd data/models
git clone https://www.modelscope.cn/l424102993/LLM_TQ_Tuned_model.git
git clone https://www.modelscope.cn/iic/nlp_bert_backbone_base_std
git clone https://www.modelscope.cn/l424102993/LLM_TQ_Regression_model.git
```

使用魔塔sdk下载(推荐)
```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('l424102993/LLM_TQ_Tuned_model', cache_dir = "./data/models/")
model_dir = snapshot_download('iic/nlp_bert_backbone_base_std', cache_dir = "./data/models/")
model_dir = snapshot_download('l424102993/LLM_TQ_Regression_model', cache_dir = "./data/models")
## 替换其他路径
```
## 使用方式

### 1. 修改配置文件

修改配置文件 ./configs/config.ymal

根据下载模型的地址修改config.ymal的模型目录以及指定cuda号

### 2. 启动评分requests服务

2.1 直接使用bass启动request服务
```bash
bash serve.py
```

2.2 手动启动
如需指定虚拟环境或其他需求, 可手动启动 ./src/evaluator_request.py
注意手动启动时,可能需要修改evaluator_request.py里config.yaml的读取的相对路径, 以及config.yaml里模型的相对路径

### 3.添加API keys
在Evaluate_example.ipynb 对应位置添加Qwen的API Keys, 或自行修改相关代码以调用其他模型

### 4.模型测试
使用Evaluate_example.ipynb 进行单个模型单次的评分测试

## 批量评估

1.使用/notebooks/Evaluate_batch.ipynb 调用api接口或本地模型进行多个模型的评分测试

2.使用/notebooks/Generated_Result_Visualization_Analysis.ipynb可视化查看并对比分析结果
