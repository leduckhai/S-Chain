import os
import argparse
import json
import pickle
def parse_option():
    parser = argparse.ArgumentParser('Processing VQA-RAD', add_help=False)
    parser.add_argument('--path', type=str, default="", help='Path to raw data files', )
    parser.add_argument('--dataset', choices=['vqa-rad', 'data_RAD', 'slake', 'pvqa'], default = 'test', help='Downstream data', )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default = 'test', help='Processing test or train dataset', )
    args = parser.parse_args()
    return args
args = parse_option()

def process_data_RAD(): 
  print('Processing data_RAD...')
  f = open(args.path)
  data = json.load(f)
  content = []
  for vqa in data:    
    #Create candidate answers for closed-set questions
    # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
    # With non-yes-no type, we have to type the candidate answers directly from command line.  
    if vqa['answer_type'] == 'CLOSED': 
      if vqa['answer'].lower() in ['yes', 'no']: 
        candidate_answers = ' (Answer with either "Yes" or "No").'
      else:
        print("'{}' is not a question with answers [yes,no] but {}".format(vqa['question'], vqa['answer']))
        print("Please type each candidate answers here. Enter new line after each candidate. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        contents =[]
        while True:
          try:
            line = input()
          except EOFError:
            break
          line = line.strip()
          contents.append(line)
        candidate_answers = ' (Answer with either {}).'.format(' or '.join(f'"{w}"' for w in contents))
        print(candidate_answers)
        print()
    else: 
      candidate_answers = ''
    #End creating candidate answers
    vqa_instruct = {}
    vqa_instruct['id'] = int(vqa['qid'])
    vqa_instruct['image'] = vqa['image_name']
    vqa_instruct['answer_type'] = vqa['answer_type']
    vqa_instruct['conversations'] = [{
      'from': 'human',
      'value': vqa['question'] + candidate_answers + '\n<image>'
      },
      {
        'from': 'gpt',
        'value': vqa['answer']
        }]
    content.append(vqa_instruct)
  file_name = 'test_w_options_new.json' if 'test' in args.path else 'train_w_options_new.json'
  with open(file_name, 'w') as f_saved: 
    json.dump(content, f_saved)
  f.close()

def process_vqa_rad(): 
  pass
def process_slake(): 
  print('Processing Slake...')
  f = open(args.path)
  data = json.load(f)
  content = []
  for vqa in data:
    if vqa['q_lang'] != 'en': 
      continue 
    #Create candidate answers for closed-set questions
    # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
    # With non-yes-no type, we have to type the candidate answers directly from command line.  
    if vqa['answer_type'] == 'CLOSED': 
      if vqa['answer'].lower() in ['yes', 'no']: 
        candidate_answers = ' (Answer with either "Yes" or "No").'
      else:
        print("'{}' is not a question with answers [yes,no] but {}".format(vqa['question'], vqa['answer']))
        print("Please type each candidate answers here. Enter new line after each candidate. Ctrl-D or Ctrl-Z ( windows ) to save it.")
        contents =[]
        while True:
          try:
            line = input()
          except EOFError:
            break
          line = line.strip()
          contents.append(line)
        candidate_answers = ' (Answer with either {}).'.format(' or '.join(f'"{w}"' for w in contents))
        print(candidate_answers)
        print()
    else: 
      candidate_answers = ''
    #End creating candidate answers
    vqa_instruct = {}
    vqa_instruct['id'] = int(vqa['qid'])
    vqa_instruct['image'] = vqa['img_name']
    vqa_instruct['answer_type'] = vqa['answer_type']
    vqa_instruct['conversations'] = [{
      'from': 'human',
      'value': vqa['question'] + candidate_answers + '\n<image>'
      },
      {
        'from': 'gpt',
        'value': vqa['answer']
        }]
    content.append(vqa_instruct)
  file_name = args.split + '_w_options_new.json'
  with open(file_name, 'w') as f_saved: 
    json.dump(content, f_saved)
  f.close()

def process_pvqa(): 
  print('Processing PathVQA...')
  with open(args.path, 'rb') as f: # args.path: /netscratch/trnguyen/pvqa/qas/train_vqa.pkl
    data = pickle.load(f)
  content = []
  for vqa in data:    
    if vqa['answer_type'] == 'yes/no': 
      candidate_answers = ' (Answer with either "Yes" or "No").'
    else: 
      candidate_answers = ''

    vqa_instruct = {}
    vqa_instruct['id'] = vqa['img_id']
    vqa_instruct['image'] = os.path.join(args.split, vqa['img_id'] + '.jpg')
    vqa_instruct['answer_type'] = 'CLOSED' if vqa['answer_type'] == 'yes/no' else 'OPEN'
    vqa_instruct['conversations'] = [{
      'from': 'human',
      'value': vqa['sent'] + candidate_answers + "\n<image>" 
      },
      {
        'from': 'gpt',
        'value': next(iter(vqa['label']))
        }]
    content.append(vqa_instruct)
  file_name = args.split + '_w_options_new.json'
  with open(file_name, 'w') as f_saved: 
    json.dump(content, f_saved)
  print('Done.')
if args.dataset == 'data_RAD': 
  process_data_RAD()
elif args.dataset == 'vqa-rad': 
  process_vqa_rad()
elif args.dataset == 'slake': 
  process_slake()
elif args.dataset == 'pvqa': 
  process_pvqa()
else: 
  raise ValueError("Unknown data, please choose from the following options: ['vqa-rad', 'data_RAD', 'slake', 'pathvqa']")


'''
Convert VQA-RAD to (one-round) conversation-style format. Dataset is saved as a list with following structure: 
[
  {
    "id": "17506892_F1",
    "image": "17506892_F1.jpg",
    "conversatons": [
      {
        "from": "human",
        "value": "<image>\nCan you describe the image for me?"
      },
      {
        "from": "gpt",
        "value": "The image consists of maps of significant voxels representing regions of hypoperfusion in FTLD patients according to their clinical diagnosis. These maps are superimposed onto a reference T1-weighted MRI image. There are five rows, each representing a different patient subgroup: bvFTD, SD, PNFA, PSP, and CBD. The neurological convention is followed, with the left side of the brain on the left side of the image."
      }
    ]
  }, 
  {
    "id":..., 
    "image": ..., 
    "conversation": ...
  
  }
'''


#Create candidate answer from all opened-set answers in training set 

#Candidate file has format: 
#{'0': [str, str, ..., str]} str: answer in training set, as string. 

'''
import json 

f = open(args.path)
data = json.load(f)
candidate_answers = {'0':[]}
for vqa in data: 
  if vqa['answer_type'] == 'OPEN' and vqa['phrase_type']  in ['freeform', 'para']: 
    answer = vqa['answer']
    if isinstance(answer, str):
      answer = answer.lower()
    if answer not in candidate_answers['0']:
      candidate_answers['0'].append(answer)
with open('candidate.json', 'w') as f_candidate: 
  json.dump(candidate_answers, f_candidate)
f.close()
'''


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Single-round conversation. Each QA is an independent element in the list. 
# Use this (not the mult-round version) for converting format of VQA-RAD TEST DATA to conversation-style
'''
import json
from num2words import num2words
f = open(args.path)
data = json.load(f)
content = []
for vqa in data:
  if args.split == 'test':
     if vqa['phrase_type'] not in ['test_freeform', 'test_para']:
        continue
  else:
     if  vqa['phrase_type'] not in ['freeform', 'para']: 
        continue
    
  #Create candidate answers for closed-set questions
  # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
  # With non-yes-no type, we have to type the candidate answers directly from command line.  
  #if vqa['answer_type'] == 'CLOSED': 
  #  if vqa['answer'].lower() in ['yes', 'no']: 
  #    candidate_answers = ' Please choose from the following two options: [yes,no]'
  #  else:
  #    print("'{}' is not a question with answers [yes,no] but {}".format(vqa['question'], vqa['answer']))
  #    print("Please type each candidate answers here. Enter new line after each candidate. Ctrl-D or Ctrl-Z ( windows ) to save it.")
  #    contents =[]
  #    while True:
  #      try:
  #        line = input()
  #      except EOFError:
  #        break
  #      line = line.strip()
  #      contents.append(line)
  #    candidate_answers = ' Please choose from the following {} options: [{}]'.format(num2words(len(contents)), ','.join(contents))
  #    print("Prompt: '{}'".format(candidate_answers))
  #    print()
  #else: 
  candidate_answers = ''
  #End creating candidate answers
  vqa_instruct = {}
  vqa_instruct['id'] = int(vqa['qid'])
  vqa_instruct['image'] = vqa['image_name']
  vqa_instruct['answer_type'] = vqa['answer_type']
  vqa_instruct['conversations'] = [{
    'from': 'human',
    'value': "<image>\n" + vqa['question'] + candidate_answers
    },
    {
      'from': 'gpt',
      'value': vqa['answer']
      }]
  content.append(vqa_instruct)
if args.split == 'train': 
  for vqa in data: 
     if vqa['question_frame']!= 'NULL': 
      vqa_instruct = {}
      vqa_instruct['id'] = int(vqa['qid'])
      vqa_instruct['image'] = vqa['image_name']
      vqa_instruct['answer_type'] = vqa['answer_type']
      vqa_instruct['conversations'] = [{
         'from': 'human',
         'value': "<image>\n" + vqa['question_frame']
         },
         {
            'from': 'gpt',
            'value': vqa['answer']
            }]
      content.append(vqa_instruct) 
file_name = 'test.json' if args.split == 'test' else 'train.json'
with open(file_name, 'w') as f_saved: 
   json.dump(content, f_saved)
 
f.close()
'''

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#Multi-round conversation. All QAs from the same image are grouped together
# Use this for converting format of VQA-RAD TRAIN DATA to conversation-style
'''
import json
from num2words import num2words
f = open(args.path)
data = json.load(f)

train = []
name_to_pos_train = {} # e.g {"synpic54610": 0, "synpic50962":1, "synpic34947":2, ... }
pos_train= 0

for vqa in data: 
  if vqa['phrase_type']  not in ['freeform', 'para']:
    continue 
  #Create candidate answers for closed-set questions
  # Most(90%) of closed-set questions are [yes, no] type, but there are also many other type of closed-set (i.e [left, right], [up, down], etc.)
  # With non-yes-no type, we have to type the candidate answers directly from command line.  
  if vqa['answer_type'] == 'CLOSED': 
    if vqa['answer'].lower() in ['yes', 'no']: 
      candidate_answers = ' Please choose from the following two options: [yes,no]'
    else:
      print("'{}' is not a question with answers [yes,no] but {}".format(vqa['question'], vqa['answer']))
      print("Please type each candidate answers here. Enter new line after each candidate. Ctrl-D or Ctrl-Z ( windows ) to save it.")
      contents =[]
      while True:
        try:
          line = input()
        except EOFError:
          break
        line = line.strip()
        contents.append(line)
      candidate_answers = ' Please choose from the following {} options: [{}]'.format(num2words(len(contents)), ','.join(contents))
      print("Prompt: '{}'".format(candidate_answers))
      print()
  else: 
    candidate_answers = ''
  #End creating candidate answers
  image_name = vqa['image_name'][:-4]
  if image_name in name_to_pos_train: 
    train[name_to_pos_train[image_name]]['conversations'].extend([{"from": "human", "value": vqa['question'] + candidate_answers}, {"from": "gpt", "value": vqa['answer']}])
  else: 
    vqa_instruct = {}
    vqa_instruct['id'] = int(vqa['qid'])
    vqa_instruct['image'] = vqa['image_name']
    vqa_instruct['conversations'] = [{
        "from": "human",
        "value": "<image>\n" + vqa['question'] + candidate_answers
      },
      {
        "from": "gpt",
        "value": vqa['answer']
      }]
    train.append(vqa_instruct)
    name_to_pos_train[image_name] = pos_train
    pos_train +=1
with open('train.json', 'w') as f_train:
    json.dump(train, f_train)
f.close()
'''


        
    
    
