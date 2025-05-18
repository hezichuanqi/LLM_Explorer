import torch
from unsloth import FastLanguageModel

# 模型配置参数
max_seq_length = 2048  # 最大序列长度
dtype = None          # 数据类型，None表示自动选择
load_in_4bit = True   # 使用4bit量化加载模型以节省显存

# 加载预训练模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    #model_name = "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    model_name = "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    local_files_only=True,  # 避免联网
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    #token = hf_token, 
)

#定义提示词模版
prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
  请写出恰当完成该请求的回答。
  在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

  ### Instruction:
  你是一位在临床推理、诊断和治疗计划方面具有专业知识的医学专家。
  请回答以下医学问题。

  ### Question:
  {}

  ### Response:
  <think>{}"""

train_prompt_style = prompt_style + """
                    </think>
                    {}"""

# 测试用医学问题
question = "一名70岁的男性患者因胸痛伴呕吐16小时就医，心电图显示下壁导联和右胸导联ST段抬高0.1~0.3mV，经补液后血压降至80/60mmHg，患者出现呼吸困难和不能平卧的症状，体检发现双肺有大量水泡音。在这种情况下，最恰当的药物处理是什么？"

#设置模型为推理模式
FastLanguageModel.for_inference(model) 
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# 生成回答
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print("### 微调前模型推理结果：")
print(response[0].split("### Response:")[1])

