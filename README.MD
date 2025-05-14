<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-05-14 01:10:37
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-05-14 02:54:36
 * @FilePath: /LLM_Scores/github/README.MD
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
## 模型下载

Qwen_7B_Review_Tuned_model : git clone https://www.modelscope.cn/l424102993/LLM_TQ_Tuned_model.git

Regression_model_base : git clone https://www.modelscope.cn/iic/nlp_bert_backbone_base_std

Regression_model_regression : git clone https://www.modelscope.cn/l424102993/LLM_TQ_Regression_model.git

```python
#SDK模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('l424102993/LLM_TQ_Tuned_model')
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