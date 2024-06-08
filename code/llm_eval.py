from egoexo_dataset import EgoExo_pose
def init_poe():
    tokens = {
        'b': 'DILCAjW1zCzFdnIs8FIQGQ%3D%3D', 
        'lat': 'jOG2v9ajZfE9Dy3FPWdwiR%2B6E0N9Ig1Ndx85GQMdhw%3D%3D'
    }
    from poe_api_wrapper import PoeApi
    client = PoeApi(cookie=tokens)

    bot = 'gpt3_5'
    prompt = "Can you estimate what kind of activity a user is performing? You will be provided with a series of action, ambient sound and other meta data. Please respond with a series of potential activities the users performed, the activties should be specific.?"
    return client, bot, prompt
def call_poe(client, bot, prompt, message):
    message = prompt + message
    for chunk in client.send_message(bot, message):
        pass
    print(chunk["text"])