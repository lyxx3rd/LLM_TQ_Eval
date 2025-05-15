import torch
from transformers import  AutoTokenizer, AutoModelForCausalLM,BertTokenizer
from regression_model import DualBertRegressionModel
from flask import Flask, request, jsonify
import yaml

# 读取 YAML 文件
with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)  # 安全加载，避免执行恶意代码

Qwen_7B_Review_Tuned_model = config['evaluator_request']['Qwen_7B_comment_Tuned_model']
Regression_model_base = config['evaluator_request']['Regression_model_base']
Regression_model_regression = config['evaluator_request']['Regression_model_regression']
calculate_device = config['evaluator_request']['calculate_device']


app = Flask(__name__)
# 1. 加载本地模型和tokenizer
model_path  = Qwen_7B_Review_Tuned_model
tokenizer = AutoTokenizer.from_pretrained(model_path)

commit_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    trust_remote_code=True
).to(calculate_device)

# 4. 初始化完整模型
regression_model = DualBertRegressionModel(Regression_model_base).to(calculate_device)
regression_tokenizer = BertTokenizer.from_pretrained(Regression_model_base)
checkpoint = torch.load(
    Regression_model_regression,
    weights_only=True
)
regression_model.load_state_dict(checkpoint['state_dict'])

instruction = "下列给出了一个用户输入的询问query, 语言模型生成的query的回复文本content请根据这两个内容生成这个content的文本质量评论, 只需给出文字的评价\n评价应从以下方面进行,只评价不足或非常到位的点, 不要泛泛而谈的内容\n内容质量:扣题程度, 专业知识准确性, 回答深度以及联想的广度, 文本逻辑性; 结构与组织:清晰度,条理性和层次感; 语言表达:语法与拼写, 冗余程度, 流畅性, 风格适配; 其他可以考虑的方面; 存在的缺点\n先给出文字评价, 对评价做出最终结论, 结论包括以下等级: 该回复属于完全不可用的回复; 该回复属于几乎不可用的回复; 该回复为不及格答案; 该回复为及格但是最基础回复; 该回复尝试深度回复但是存在明显短缺; 该回复尝试深度回复但是存在一定程度的缺漏或问题; 该回复几乎没有明显缺陷; 该回复几乎没有缺陷且存在明显闪光点.\n参考标准: 内容重复逻辑混乱但内容与有query关为不及格答案; 存在严重错误也为不及格; 结构清晰但内容泛泛为及格但是最基础回复; 虽然主题相关但未完全扣紧主题且部分内容冗余为及格但是最基础回复.\n输出格式为:<reasoning>文本评价</reasoning>\n<ans>结论等级</ans>"

# 3. 切换到评估模式
regression_model.eval()
commit_model.eval()

def predict(reasoning, ans , content):
    """
    使用BERT回归模型预测0-100之间的数值
    
    参数:
        text: 输入文本(长度不超过512)
        regression_model: 训练好的BERT回归模型
        tokenizer: BERT tokenizer
        device: 使用的设备(cuda/cpu)
        
    返回:
        预测值(0-100之间的数值)
    """
    # 预处理文本 - BERT不需要对话模板，直接处理原始文本
    inputs_reasoning = regression_tokenizer(
        reasoning,
        padding='max_length',  # 填充到最大长度
        truncation=True,       # 截断到最大长度
        max_length=512,       # BERT的最大长度
        return_tensors='pt'    # 返回PyTorch张量
    ).to(calculate_device)

    inputs_ans = regression_tokenizer(
        ans,
        padding='max_length',  # 填充到最大长度
        truncation=True,       # 截断到最大长度
        max_length=512,       # BERT的最大长度
        return_tensors='pt'    # 返回PyTorch张量
    ).to(calculate_device)

    inputs_content = regression_tokenizer(
        content,
        padding='max_length',  # 填充到最大长度
        truncation=True,       # 截断到最大长度
        max_length=512,       # BERT的最大长度
        return_tensors='pt'    # 返回PyTorch张量
    ).to(calculate_device)
    
    # 预测
    with torch.no_grad():
        prediction = regression_model(
            reasoning_ids = inputs_reasoning['input_ids'], ans_ids = inputs_ans['input_ids'], content_ids = inputs_content['input_ids'],
            reasoning_mask = inputs_reasoning['attention_mask'],ans_mask = inputs_ans['attention_mask'], content_mask = inputs_content['attention_mask']
        )
    
    regression_model.train()
    return prediction.item()


def chat_transformers_model(content, system_content="你是一个有用的助手"):
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(commit_model.device)

    generated_ids = commit_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def test_model():
    ans = predict("你好","你好","你好")
    print(ans)
    ans = chat_transformers_model("你好")
    print(ans)

# 第一个应用实例的路由
@app.route('/evaluate', methods=['POST'])
def chat_05B():
    #print("开始")
    data = request.json
    #print("接受数据")
    query = data['query']
    content = data['content']
    #print("输入成功")
    
    input_content = f'<query>:{query}</query>\n\n<content>:{content}</content>'
    instruction_input_content = f"{instruction}\n{input_content}"
    text = chat_transformers_model(instruction_input_content)
    #print("输出成功,len:",len(text))
    reasoning_strat = text.find("<reasoning>:")
    reasoning_end = text.find("</reasoning>")
    ans_start = text.find("<ans>:")
    ans_end = text.find("</ans>")
    reasoning = text[reasoning_strat+len("<reasoning>:"):reasoning_end]
    ans = text[ans_start+len("<ans>:"):ans_end]
    #print("切分成功!")
    score = predict(reasoning,ans,input_content) *100
    #print("score:",score)   

    return jsonify({"reasoning": reasoning,
                    "ans":ans,
                    "score":score})


if __name__ == '__main__':
    # 第一个应用实例监听 5000 端口
    test_model()
    app.run(host='0.0.0.0', port=4399)