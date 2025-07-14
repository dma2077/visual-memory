import json

def load_jsonl(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            results.append(line)
    return results

def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

path = '/map-vepfs/dehua/code/visual-memory/answers/food2k/food2k_results_softmax.jsonl'
results = load_jsonl(path)

results.sort(key=lambda x: int(x['image_path'].split('/')[-2]))

print(results[0])

output_path = '/map-vepfs/dehua/code/visual-memory/answers/food2k/sorted_food2k_results_softmax.jsonl'
write_jsonl(output_path, results)