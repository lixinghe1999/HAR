from transformers import AutoModelForCausalLM, AutoTokenizer
import time
device = "cuda"
 
def load_model():
    model_name = "Qwen/Qwen2-7B-Instruct"
    # model_name = "Qwen/Qwen2-7B-Instruct-GPTQ-Int8"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        cache_dir = './cache'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
def inference(model, tokenizer, prompt="You are a helpful assistant.", message="Give me a short introduction to large language model."):
    # message = "Give me a short introduction to large language model."
    # prompt = "You are a helpful assistant."
    t_start = time.time()
    if type(message) == str:
        m = message
    else:
        m = "\n".join(message)
    
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": m}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    texts = [text]
    model_inputs = tokenizer(texts, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print('inference latency', time.time() - t_start)
    return response
if __name__ == '__main__':
    model, tokenizer = load_model()
    prompt = "As a helpful AI assistant, you will be provided with multiple senstences describing an action. If the action can produce sound, \
    you will generate the objects that would create that sound. For example, if the action is 'walking', I would generate 'feet-ground'. \
    If there are additional details provided about the action, you will include those as well like 'sports shoes-gym floor'.\
    If the action described cannot produce any sound, you will simply state 'no sound'.\
    For each sentence, please list all the possibilities"
    message=["The subject is running to pass the football by left foot.", "The subject is running to pass the football by right foot."]
    for i in range(1):
        response = inference(model, tokenizer, prompt, message)
        # response = response.split('\n')
        print(response)
        print(len(response), type(response))
    