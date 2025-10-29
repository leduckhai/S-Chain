import random
import json
with open('/netscratch/trnguyen/data_RAD/train_w_options_new_full.json', 'r') as f:
    data = json.load(f)
full_size = len(data)
print(full_size)
random.shuffle(data)
val = data[:451]
train = data[451:]
with open('/netscratch/trnguyen/data_RAD/train_w_options_new.json', 'w') as f: 
    json.dump(train, f)
with open('/netscratch/trnguyen/data_RAD/val_w_options_new.json', 'w') as f: 
    json.dump(val, f)
