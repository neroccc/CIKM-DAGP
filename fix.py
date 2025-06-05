import json

valid_labels = {"Easy", "Medium", "Hard"}
input_file = "leetcode-train.jsonl"
output_file = "leetcode-train-clean.jsonl"
cleaned_data = []
with open(input_file, 'r') as fin:
    for line in fin:
        try:
            entry = json.loads(line)
            if 'difficulty' in entry and entry['difficulty'] in valid_labels:
                cleaned_entry = {
                    'content': entry.get('content', ''),
                    'difficulty': entry['difficulty'],
                    'id': entry.get('id', -1)
                }
                cleaned_data.append(cleaned_entry)
        except json.JSONDecodeError:
            continue  # skip malformed lines

with open(output_file, 'w') as fout:
    for item in cleaned_data:
        fout.write(json.dumps(item) + "\n")
