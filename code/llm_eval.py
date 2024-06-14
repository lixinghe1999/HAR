from egoexo_dataset import EgoExo_atomic
def init_poe():
    tokens = {
        'b': 'DILCAjW1zCzFdnIs8FIQGQ%3D%3D', 
        'lat': 'jOG2v9ajZfE9Dy3FPWdwiR%2B6E0N9Ig1Ndx85GQMdhw%3D%3D'
    }
    from poe_api_wrapper import PoeApi
    client = PoeApi(cookie=tokens)

    bot = 'gpt3_5'
    prompt = "Can you estimate what kind of activity a user is performing based the ambient sound? You will be provided the format like{'start': 10.0, 'end': 11.0, 'tags': [{'idx': the index of sound event, 'label': the name of sound event, 'probability'}]}: . Please respond with a series of potential activities the users performed, the activties should be specific.?"
    return client, bot, prompt
def call_poe(client, bot, prompt, message):
    message = prompt + message
    for chunk in client.send_message(bot, message):
        pass
    print(chunk["text"])

if __name__ == "__main__":
    dataset = EgoExo_atomic(window_sec=4, modal=['efficientAT'])
    data = dataset[0]
    print(data)
    # client, bot, prompt = init_poe()
    # message = "music"
    # call_poe(client, bot, prompt, message)