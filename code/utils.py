import json
import math
import torch
import torch.nn.functional as nnf

def save_jsonl(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
            
def add_jsonl(filename, data):
    with open(filename, 'a+') as f:
        f.write(json.dumps(data) + '\n')


def load_jsonl(filename):
    results = []
    with open(filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            results.append(line)
    return results

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def save_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nnf.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)

def from_path2category(datasetname, path, label_dict):
    if datasetname == 'food101' or datasetname == 'veg200' or datasetname == 'foodx251':
        category = path.split('/')[-2].replace('_', ' ')
    elif datasetname == 'food172':
        category_id_1 = path.split('/')[-2]
        category_id = int(category_id_1) - 1
        category = label_dict[category_id]
    elif datasetname == 'food2k':
        category_id = path.split('/')[-2]
        category = label_dict[int(category_id)]
    elif datasetname == 'fru92':
        category = path.split('/')[-2].replace('_', ' ')
    else:
        raise ValueError(f"Dataset {datasetname} not supported")    
    return category