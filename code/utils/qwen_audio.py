from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True, cache_dir='./cache').eval()
    return model, tokenizer
def inference(model, tokenizer):
    # 1st dialogue turn
    query = tokenizer.from_list_format([
        {'audio': 'tmp.flac'}, # Either a local path or an url
        {'text': 'Describe the content of the audio.'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    # print(response)
    # The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".

    # 2nd dialogue turn
    # response, history = model.chat(tokenizer, 'Find the start time and end time of the word "middle classes"', history=history)
    # print(response)
    # The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
    return response