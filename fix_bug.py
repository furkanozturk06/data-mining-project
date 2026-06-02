import json
with open('04_bert_finetuning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    new_source = []
    for line in cell.get('source', []):
        line = line.replace("device == 'cuda'", "str(device).startswith('cuda')")
        line = line.replace("device == \"cuda\"", "str(device).startswith('cuda')")
        new_source.append(line)
    cell['source'] = new_source

with open('04_bert_finetuning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
