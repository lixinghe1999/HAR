# from utils.qwen2 import load_model, inference

# model, tokenizer = load_model()
# prompt = "As a helpful AI assistant, you will be provided with multiple senstences describing an action. If the action can produce sound, \
# you will generate the objects that would create that sound. For example, if the action is 'walking', I would generate 'feet-ground'. \
# If there are additional details provided about the action, you will include those as well like 'sports shoes-gym floor'.\
# If the action described cannot produce any sound, you will simply state 'no sound'.\
# For each sentence, please list all the possibilities"
# message=["The subject is running to pass the football by left foot.", "The subject is running to pass the football by right foot."]
# for i in range(1):
#     response = inference(model, tokenizer, prompt, message)
#     # response = response.split('\n')
#     print(response)
#     print(len(response), type(response))


# dataset = Ego4D_Understanding(folder='../dataset/ego4d/v2/', window_sec=20, modal=['imu', 'audio'])
# data = dataset[0]