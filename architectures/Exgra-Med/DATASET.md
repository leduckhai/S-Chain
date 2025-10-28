<div align='center'>

# DATASET

</div>

This file summarizes the structure of all datasets used in our project, as well as problems and solutions to fix them. 
## Downstream dataset 
In LLaVA-Med, little information is provided regarding the preparation of downstream dataset (VQA-RAD, PathVQA and Slake). Here we present the way to prepare them. 

After the corresponding dataset is downloaded following instruction from its homepage, three components are needed: 
1. `train.json`: a json file containing all question-answer pairs (and corresponding image information) used for training
2. `test.json`: a json file containing all question-answer pairs (and correponding image information) used for evaluation
3. `images`: folder contains all images 

The test and train file in general have the same structure, which must have the following form (the field in the angle bracket '<>' is filled according to each QA and dataset): 
```
[
    {
        "id": <image_ID> # str or int, depends on dataset, could be the name of image
        "image": <image_name> # str, as saved in 'images' folder, including extension (jpg, png, etc.)
        "answer_type": "CLOSED" | "OPEN" # Indicate this QA is a closed- (yes/no) or open-set
        "conversations": [
            {"from": "human", 
             "value": <question1> # str},
            {"from": "gpt",
             "value": <answer1> # str}, 
            {"from": "human", 
             "value": <question2>},
            {"from": "gpt",
             "value": <answer2>}, 
                     ...
            {"from": "human", 
             "value": <question_n>},
            {"from": "gpt",
             "value": <answer_n>}
        ]
    }, 
    {
        "id":...
        ...
    }
]
```
There should be a string `"<image>\n"` included in `<question1>`, which indicates that image is also included as token together with the text tokens (imagine our first question in the conversation is actually text + image). For closed-set QAs, the candidate answers must be provided in the question (see paper for more details). We decided to use the following phrase to encode candidate answers into question: `"Please choose from the following two options: [yes, no]"`. For example, if our QA pair is `Q: Are the lungs normal appearing?, A: no`, then `<question1>` is `"<image>\nAre the lungs normal appearing? Please choose from the following two options: [yes, no]"` (omit `"<image>\n"` if it is not the 1st question in conversation) and `<answer1>` is `"no"`.For open-set QA there is no need to add this phrase. 

Please note that about 10% closed-set QA question in VQA-RAD does not have yes/no or have more than two candidate answers, for example `Q: Is this image normal or abnormal?, A: normal` (this should be similar for other dataset). One might just ignore these QAs and add the standard yes/no phrase to their questions. However, to increase the accuracy, we decided to look into them as well. We also first add the standard yes/no phrase to all questions, then for closed-set questions without yes/no answer, we add the answer candidates manually via command line. A script for automating this process is found in /netscratch/trnguyen/VQA-RAD/processing.py. 

The format structure above is of `train.json`. For test set, the format looks exactly the same, except that in `test.json` the`"conversations":[]` is only one-round, i.e contains only one question and answer. The reason is we need to consider each QA seperately to classify them into closed- or open-set and thus have the appropriate metrics to measure their perfomance. 

### About the candidate file

Candidate file is only needed for discriminative VQA assistant, not our model (which is generative). During evaluation on test set, the predicted answer is compared with each answer from the training set, and the answer with the highest similarity score with predicted answer is considered the true predicted result, which is then compared with GT answer. So the candidate file is just a accumulation of all answers in the training set. LLaVA-Med only cares about accuracy and recall, so no need to use this candidate file. Because the evaluation script still requires this file, one can just simply delete every lines of code that refer to it. However, if one wants to use it, please see /netscratch/trnguyen/VQA-RAD/processing.py for how to generate candidate file. 
