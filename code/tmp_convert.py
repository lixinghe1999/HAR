import json
input_json = 'resources/egoexo_atomic_1.json'
output_json = 'resources/egoexo_atomic.json'
json_dataset = json.load(open(input_json))
keys = list(json_dataset.keys())
num_samples = len(json_dataset[keys[0]])    
new_json = []
for i in range(num_samples):
    new_json.append({})
    for key in keys:
        new_json[i][key] = json_dataset[key][i]
print(new_json[0])
json.dump(new_json, open(output_json, 'w'), indent=4)