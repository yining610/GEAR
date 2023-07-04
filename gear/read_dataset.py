import xml.etree.ElementTree as ET
import json
import os
import re
import numpy as np

def read_ASDiv(path_to_ASDiv):
    """
    example of path_to_ASDiv:
        "../datasets/ASDiv/ASDiv.xml"
    """

    inputs, gold_answers, question_ids = [], [], []
    tree = ET.parse(path_to_ASDiv)
    root = tree.getroot()
    for problem in root.findall('Problem'):
        question = problem.find('Body').text + " " + problem.find('Question').text
        gold_answer = problem.find('Answer').text
        question_id = problem.attrib['ID']

        inputs.append(question)
        gold_answers.append(gold_answer)
        question_ids.append(question_id)
    
    return inputs, gold_answers, question_ids

def read_IWSLT(src_file):
    """
    example of src_file: 
        "../datasets/IWSLT/en-zh/IWSLT17.TED.tst2015.en-zh.zh.xml"
    """
    # read IWSLT2017 xml file
    inputs, gold_answers, question_ids = [], [], []

    # Non-English to English
    source_langauge = src_file.split('.')[-2] + ".xml"

    target_language = "en" + ".xml"

    target_file = src_file.replace(source_langauge, target_language)

    tree_src = ET.parse(src_file)
    tree_target = ET.parse(target_file)

    root_src = tree_src.getroot()[0]
    root_target = tree_target.getroot()[0]

    # prefix
    prefix = ['How to say ', "What is ", "Translate ", "", "Speak ", "Express "]
    suffix = [' in English?', ' in English.', ' to English?', ' to English.']

    for doc_src, doc_target in zip(root_src.findall('doc'), root_target.findall('doc')):
        doc_id = doc_src.attrib['docid']
        assert doc_id == doc_target.attrib['docid']
        for src_lan, tar_lang in zip(doc_src.findall('seg'), doc_target.findall('seg')):
            seg_id = src_lan.attrib['id']
            assert seg_id == tar_lang.attrib['id']

            # random sample a prefix and a sufix
            prefix_idx = np.random.randint(0, len(prefix))
            suffix_idx = np.random.randint(0, len(suffix))

            input = prefix[prefix_idx] + src_lan.text.strip() + suffix[suffix_idx]
            gold_answer = tar_lang.text.strip()
            id = doc_id + "_" + seg_id

            inputs.append(input)
            gold_answers.append(gold_answer)
            question_ids.append(id)

    return inputs, gold_answers, question_ids

def read_MLQA(path_to_MLQA):
    """
    example of path_to_MLQA: 
        "../datasets/MLQA_V1/context-en-question-ar/"
    """

    inputs, gold_answers, question_ids = [], [], []

    # path_to_MLQA is a directory
    for file in os.listdir(path_to_MLQA):
        with open(os.path.join(path_to_MLQA, file), "r") as f:
            data = json.load(f)
            for i in range(len(data["data"])):                       
                context = data["data"][i]['paragraphs'][0]['context']
                question = data["data"][i]['paragraphs'][0]['qas'][0]['question']
                answer = data["data"][i]['paragraphs'][0]['qas'][0]['answers'][0]['text']
                id = data["data"][i]['paragraphs'][0]['qas'][0]['id']
                input = f"question: {question} context: {context}"
                
                inputs.append(input)
                gold_answers.append(answer)
                question_ids.append(id)
    
    return inputs, gold_answers, question_ids


def read_SVAMP(path_to_SVAMP):
    """
    example of path_to_SVAMP:
        "../datasets/SVAMP/SVAMP.json"
    """

    inputs, gold_answers, question_ids = [], [], []

    with open(path_to_SVAMP, "r") as f:
        data = json.load(f)
    for item in data:
        context = item["Body"]
        question = item["Question"]
        gold_answer = item['Answer']
        id = item['ID']
        input = context + " " + question
        
        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(id)

    return inputs, gold_answers, question_ids

def read_GSM8K(path_to_GSM8K):
    """
    example of path_to_GSM8K:
        "../datasets/GSM8K/GSM8K_test.jsonl"
    """

    inputs, gold_answers, question_ids = [], [], []

    with open(path_to_GSM8K, "r") as f:
        data = f.readlines()
    for idx, item in enumerate(data):
        item = json.loads(item)
        input = item["question"]
        # find the answer after ####
        gold_answer = item['answer'].split("####")[1]
        
        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(idx)

    return inputs, gold_answers, question_ids

def read_TriviaQA(path_to_triviaqa):
    """
    example of path_to_triviaqa:
        "../datasets/TriviaQA/TriviaQA1k.json"
    """

    with open(path_to_triviaqa, "r") as f:
        data = json.load(f)
    
    inputs, gold_answers, question_ids = [], [], []
    for item in data:
        id = item["QuestionId"]
        NormalizedValue = item["Answer"]["NormalizedValue"]
        NormalizedAliases = item["Answer"]["NormalizedAliases"]

        NormalizedAliases.append(NormalizedValue)

        gold_answer = list(set(NormalizedAliases))

        input = item["Question"]

        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(id)

    return inputs, gold_answers, question_ids


def read_squad(path_to_squad):
    """
    example of path_to_squad:
        "../datasets/SQuAD/squad.json"
    """
    
    with open(path_to_squad, "r") as f:
        data = json.load(f)

    inputs, gold_answers, question_ids = [], [], []
    for item in data:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                if qa["is_impossible"]:
                    continue
                question = qa["question"]
                id = qa["id"]
                input = "Question: " + question + " Context: " + context
                gold_answer = qa["answers"][0]["text"]

                inputs.append(input)
                gold_answers.append(gold_answer)
                question_ids.append(id)

    return inputs, gold_answers, question_ids


def read_webqs(path_to_webqs):
    """
    example of path_to_webqs:
        "../datasets/WebQS/WebQS1k.json"
    """

    with open(path_to_webqs, "r") as f:
        data = json.load(f)
    
    inputs, gold_answers, question_ids = [], [], []
    for item in data:
        id = item['url']
        question = item['utterance']
        gold_answer = item['targetValue']
    
        inputs.append(question)
        gold_answers.append(gold_answer)
        question_ids.append(id)
    
    return inputs, gold_answers, question_ids

def read_nq_open(path_to_nqopen):
    """
    example of path_to_nqopen:
        "../datasets/NQ-Open/NQ-Open1k.json"
    """

    inputs, gold_answers, question_ids = [], [], []

    with open(path_to_nqopen, "r") as f:
        data = json.load(f)
    for idx, item in enumerate(data):
        input = item["question"]
        gold_answer = item["answer"]
        id = idx

        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(id)


    return inputs, gold_answers, question_ids


def read_trex(path_to_trex):
    inputs, gold_answers, question_ids = [], [], []
    with open(path_to_trex, "r") as f:
        data = json.load(f)

    for item in data:
        input = f"What is {item['title']}?"
        gold_answer = item['text']
        id = item['docid']

        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(id)

    return inputs, gold_answers, question_ids


    


def read_commonsenseQA(path_to_commonsenseqa):
    """
    example of paht_to_commonsenseqa:
        "../datasets/CommonsenseQA/commonsenseQA1k.json"
    """

    inputs, gold_answers, question_ids = [], [], []
    with open(path_to_commonsenseqa, "r") as f:
        data = json.load(f)
    
    for item in data:
        input = item["question"]["stem"] + " Choices: "
        gold_answer = item["answerKey"] + ": " + item["question"]["choices"][ord(item["answerKey"]) - ord("A")]["text"]
        id = item["id"]

        for choice in item["question"]["choices"]:
            input = input + " " + choice['label'] + ": " + choice["text"]
        
        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(id)
    
    return inputs, gold_answers, question_ids

def read_copa(path_to_copa):
    """
    example of path_to_copa:
        "../datasets/COPA/copa1k.json"
    """

    inputs, gold_answers, question_ids = [], [], []
    with open(path_to_copa, "r") as f:
        data = json.load(f)
    
    for idx, item in enumerate(data):
        input = item["premise"] + " " + item["question"] + "?" + " Choice1: " + item["choice1"] + " Choice2: " + item["choice2"]
        gold_answer = "Choice1: " + item["choice1"] if item["label"] == 1 else "Choice2: " + item["choice2"] 
        
        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(idx)
    
    return inputs, gold_answers, question_ids
    
def read_timezone(path_to_timezonefile):
    """
    example of path_to_timezonefile:
        "../datasets/TimeZone/timezone.json"
    """

    inputs, gold_answers, question_ids = [], [], []
    with open(path_to_timezonefile, "r") as f:
        data = json.load(f)
    
    for item in data:
        input = item["text"]
        gold_answer = item["gold_answer"]
        id = item["id"]

        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(id)
    
    return inputs, gold_answers, question_ids
        

def read_socialiqa(path_to_socialiqa):
    """
    example of path_to_socialiqa:
        "../datasets/SocialIQA/socialiqa1k.json"
    """

    inputs, gold_answers, question_ids = [], [], []
    with open(path_to_socialiqa, "r") as f:
        data = json.load(f)
    
    for idx, item in enumerate(data):
        input = item['context'] + " " + item['question'] + " Choice1: " + item['answerA'] + " Choice2: " + item['answerB'] + " Choice3: " + item['answerC']
        gold_answer = "Choice1: " + item['answerA'] if item['label'] == "1" else "Choice2: " + item['answerB'] if item['label'] == "2" else "Choice3: " + item['answerC']


        inputs.append(input)
        gold_answers.append(gold_answer)
        question_ids.append(idx)

        # print(input)
        # print(gold_answer)
        # print(idx)

    return inputs, gold_answers, question_ids

