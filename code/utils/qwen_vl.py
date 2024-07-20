from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
def init_qwenvl():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, cache_dir='./cache').eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
    return model, tokenizer
def inference_qwenvl(model, tokenizer):
    # 1st dialogue turn
    query = tokenizer.from_list_format([
        # {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, # Either a local path or an url
        # {'text': '这是什么?'},
        {'image': 'tmp.jpg'},
        {'text': 'This is an egocentric image taken by a smart glass, please describe the action of the user.'},
    ])    
    response, history = model.chat(tokenizer, query=query, history=None)
    return response