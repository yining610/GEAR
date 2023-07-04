import pandas as pd
import numpy as np
import re
from nltk.translate.bleu_score import sentence_bleu
import math

def _text2int(text: str, numwords={}) -> str:
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]
      tens = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
      scales = ["hundred", "thousand", "million", "billion", "trillion"]
    #   numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, (idx+2) * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
    current = result = 0
    output = ""
    # split by space or comma or dot
    tokens = re.split("\s|(?<!\d)[,.:](?!\d)", text)
    for idx, word in enumerate(tokens):
        if word not in numwords:
            num = result + current
            if num != 0:
                output = output + str(num) + " "
            result = current = 0
            output = output + word + " "
        elif idx != len(tokens) - 1:
            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        else: 
            scale, increment = numwords[word]
            current = current * scale + increment
            result += current
            output = output + str(result)
    return output

def _process_potential_answers(potential_answer: str) -> str:
    """
    Process the potential answer by removing the \n and stripping. Converting numeric words to numbers
    Args:
        potential_answer (str): the potential answer
    Returns:
        str: the processed potential answer
    """
    potential_answer = potential_answer.replace("\n", " ")
    potential_answer = potential_answer.strip().lower()
    potential_answer = _text2int(potential_answer) # convert numeric words to numbers
    return potential_answer

# def extract_number(text: str):
#     """Extract the first number from a given text."""
#     if text is None:
#         return None
#     finding = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
#     if(len(finding) > 1):
#         print(text)
#         print(finding)
#         print()
#     if len(finding) > 0:
#         return float(finding[0])
#     else:
#         return None

def match_number(text: str, answer: str):
    if text is None:
        return False
    finding = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if len(finding) == 0:
        return False
    
    goldanswer = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", answer)
    
    if len(goldanswer) == 0:
        return False

    # for f in finding:
    #     if float(f) == float(goldanswer[0]):
    #         return True

    f = float(finding[-1])
    if f == float(goldanswer[0]):
        return True
    return False
    

    # return re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)[0] if len(re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)) > 0 else None

def get_filtered_api(row, filter_api="filtered_apis"):
    if row[filter_api] is None:
        return None
    if type(row[filter_api]) == float:
        return None
    if type(row[filter_api]) == list or len(row[filter_api]) == 0:
        return None
    return list(row[filter_api].keys())[0]

def calculate_bleu(text, ref):
    if text is None:
        return 0
    else:
        text_split = text.strip().lower().split()
        ref_split = ref.strip().lower().split()
        return sentence_bleu([ref_split], text_split, weights = (1, 0))

process_potential_answers = lambda x: _process_potential_answers(x) if x is not None else None

def match_answer(gold_answer, answer, check_percentage=False):
    if gold_answer is None or answer is None:
        return 0

    answer_seg = answer.strip().lower().split()
    highest_match = 0
    if isinstance(gold_answer, str):
        gold_answer = [gold_answer]
    for gold_answer_sub in gold_answer:
        matched = 0
        gold_seg = gold_answer_sub.strip().lower().split()
        for word in gold_seg:
            if word in answer_seg:
                if not check_percentage:
                    return 1
                else:
                    matched += 1
        match_rate = matched / len(gold_seg)
        if match_rate > highest_match:
            highest_match = match_rate
    if not check_percentage:
        return 0
    else:
        return highest_match
    
def match_multiplechoice_answer(answer, text):
    if text is None:
        return False
    answer_label = answer.split(":")[0].strip().lower()
    if answer_label == "choice1":
        answer_label = "a"
    elif answer_label == "choice2":
        answer_label = "b"
    elif answer_label == "choice3":
        answer_label = "c"
    elif answer_label == "choice4":
        answer_label = "d"
    answer_text = answer.split(":")[1].strip().lower().replace(" ", "")

    texts = text.split(":")
    text_label = texts[0].strip().lower().split(" ")[-1].strip().lower()
    if text_label == "choice1":
        text_label = "a"
    elif text_label == "choice2":
        text_label = "b"
    elif text_label == "choice3":
        text_label = "c"
    elif text_label == "choice4":
        text_label = "d"
    elif text_label == "1":
        text_label = "a"
    elif text_label == "2":
        text_label = "b"
    elif text_label == "3":
        text_label = "c"
    elif text_label == "4":
        text_label = "d"
    
    if answer_label == text_label:
        return True
    
    if answer_text in text.strip().lower().replace(" ", ""):
        return True

    return False

def read_math_dataset(dataframe, artdataframe=None, artdataframe_chatgpt=None):
    select_accuracy = dataframe.apply(lambda x: get_filtered_api(x) == "Calculator", axis=1).sum() / len(dataframe)

    select_accuracy_medium = None
    select_accuracy_large = None
    select_accuracy_xl = None
    select_accuracy_art = None
    select_accuracy_chatgpt = None
    select_accuracy_10apis = None

    if "filtered_apis_medium" in dataframe.columns:
        select_accuracy_medium = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_medium") == "Calculator", axis=1).sum() / len(dataframe[dataframe["filtered_apis_medium"].notna()])

    if "filtered_apis_large" in dataframe.columns:
        select_accuracy_large = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_large") == "Calculator", axis=1).sum() / len(dataframe[dataframe["filtered_apis_large"].notna()])
    
    if "filtered_apis_xl" in dataframe.columns:
        select_accuracy_xl = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_xl") == "Calculator", axis=1).sum() / len(dataframe[dataframe["filtered_apis_xl"].notna()])
        
    if "filtered_10apis" in dataframe.columns:
        select_accuracy_10apis = dataframe.apply(lambda x: get_filtered_api(x, "filtered_10apis") == "Calculator", axis=1).sum() / len(dataframe[dataframe["filtered_10apis"].notna()])

    pure_chatgpt = dataframe[dataframe["selected_api_chatgpt"] == "Calculator"]
    pure_gptj = dataframe[dataframe["selected_api_gptj"] == "Calculator"]

    if artdataframe is not None:
        select_accuracy_art = artdataframe.iloc[0]["selection_ratio"]
        pure_art_gptj = artdataframe[artdataframe["task"] == "Math"]
        pure_art_chatgpt = artdataframe_chatgpt[artdataframe_chatgpt['task'] == 'Math']

    gptj_accuracy = None
    pure_gptj_accuracy = None
    chatgpt_accuracy = None
    art_gptj_accuracy = None
    pure_art_gptj_accuracy = None
    pure_chatgpt_accuracy = None
    chatgpt_zeroshot_accuracy = None
    gptj_fewshot_accuracy = None
    chatgpt_fewshot_accuracy = None
    gptj_zeroshot_accuracy = None
    art_chatgpt_accuracy = None
    pure_art_chatgpt_accuracy = None

    if artdataframe is not None:
        art_gptj_accuracy = artdataframe.apply(lambda x: match_number(x["answer"], x["gold_answer"]), axis=1).sum() / len(artdataframe)
        pure_art_gptj_accuracy = pure_art_gptj.apply(lambda x: match_number(x["answer"], x["gold_answer"]), axis=1).sum() / len(pure_art_gptj)
        art_chatgpt_accuracy = artdataframe_chatgpt.apply(lambda x: match_number(x["answer"], x["gold_answer"]), axis=1).sum() / len(artdataframe_chatgpt)
        pure_art_chatgpt_accuracy = pure_art_chatgpt.apply(lambda x: match_number(x['answer'], x['gold_answer']), axis=1).sum() / len(pure_art_chatgpt)

    if "answer_gptj" in dataframe.columns:
        gptj_accuracy = (dataframe.apply(lambda x: match_number(x["answer_gptj"], x["gold_answer"]), axis=1)).sum() / len(dataframe)
        pure_gptj_accuracy = (pure_gptj.apply(lambda x: match_number(x["answer_gptj"], x["gold_answer"]), axis=1)).sum() / len(pure_gptj)

    if "answer_chatgpt" in dataframe.columns:
        chatgpt_accuracy = (dataframe.apply(lambda x: match_number(x["answer_chatgpt"], x["gold_answer"]), axis=1)).sum() / len(dataframe)
        pure_chatgpt_accuracy = (pure_chatgpt.apply(lambda x: match_number(x["answer_chatgpt"], x["gold_answer"]), axis=1)).sum() / len(pure_chatgpt)

    if "answer_chatgpt_zero_shot" in dataframe.columns:
        chatgpt_zeroshot_accuracy = (dataframe.apply(lambda x: match_number(x["answer_chatgpt_zero_shot"], x["gold_answer"]), axis=1)).sum() / len(dataframe)

    if "answer_gptj_zero_shot" in dataframe.columns:
        gptj_zeroshot_accuracy = (dataframe.apply(lambda x: match_number(x["answer_gptj_zero_shot"], x["gold_answer"]), axis=1)).sum() / len(dataframe)

    if "answer_gptj_few_shot" in dataframe.columns:
        gptj_fewshot_accuracy = (dataframe.apply(lambda x: match_number(x["answer_gptj_few_shot"], x["gold_answer"]), axis=1)).sum() / len(dataframe)

    if "answer_chatgpt_few_shot" in dataframe.columns:
        chatgpt_fewshot_accuracy = (dataframe.apply(lambda x: match_number(x["answer_chatgpt_few_shot"], x["gold_answer"]), axis=1)).sum() / len(dataframe)

    return {
        "select_accuracy": select_accuracy,
        "select_accuracy_medium": select_accuracy_medium,
        "select_accuracy_large": select_accuracy_large,
        "select_accuracy_xl": select_accuracy_xl,
        "select_accuracy_art": select_accuracy_art,
        "select_accuracy_10apis": select_accuracy_10apis,
        "gptj_accuracy": gptj_accuracy,
        "pure_gptj_accuracy": pure_gptj_accuracy,
        "art_gptj_accuracy": art_gptj_accuracy,
        "pure_art_gptj_accuracy": pure_art_gptj_accuracy,
        "chatgpt_accuracy": chatgpt_accuracy,
        "pure_chatgpt_accuracy": pure_chatgpt_accuracy,
        'art_chatgpt_accuracy': art_chatgpt_accuracy,
        'pure_art_chatgpt_accuracy': pure_art_chatgpt_accuracy,
        "chatgpt_zeroshot_accuracy": chatgpt_zeroshot_accuracy,
        "gptj_zeroshot_accuracy": gptj_zeroshot_accuracy,
        "gptj_fewshot_accuracy": gptj_fewshot_accuracy,
        "chatgpt_fewshot_accuracy": chatgpt_fewshot_accuracy
    }

def read_mt_dataset(dataframe, artdataframe=None, artdataframe_chatgpt=None):
    select_accuracy = dataframe.apply(lambda x: get_filtered_api(x) == "MT", axis=1).sum() / len(dataframe)

    select_accuracy_medium = None
    select_accuracy_large = None
    select_accuracy_xl = None
    select_accuracy_art = None
    select_accuracy_10apis = None

    if "filtered_apis_medium" in dataframe.columns:
        dataframe["filtered_apis_medium"].replace([], np.nan, inplace=True)
        select_accuracy_medium = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_medium") == "MT", axis=1).sum() / len(dataframe[dataframe["filtered_apis_medium"].notna()])

    if "filtered_apis_large" in dataframe.columns:
        dataframe["filtered_apis_large"].replace([], np.nan, inplace=True)
        select_accuracy_large = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_large") == "MT", axis=1).sum() / len(dataframe[dataframe["filtered_apis_large"].notna()])

    if "filtered_apis_xl" in dataframe.columns:
        dataframe["filtered_apis_xl"].replace([], np.nan, inplace=True)
        select_accuracy_xl = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_xl") == "MT", axis=1).sum() / len(dataframe[dataframe["filtered_apis_xl"].notna()])

    if "filtered_10apis" in dataframe.columns:
        dataframe["filtered_10apis"].replace([], np.nan, inplace=True)
        select_accuracy_10apis = dataframe.apply(lambda x: get_filtered_api(x, "filtered_10apis") == "MT", axis=1).sum() / len(dataframe[dataframe["filtered_10apis"].notna()])

    if artdataframe is not None:
        select_accuracy_art = artdataframe.iloc[0]["selection_ratio"]
        pure_art_gptj = artdataframe[artdataframe["task"] == "mt"]
        pure_art_chatgpt = artdataframe_chatgpt[artdataframe_chatgpt['task'] == 'mt']

    

    pure_chatgpt = dataframe[dataframe["selected_api_chatgpt"] == "MT"]
    pure_gptj = dataframe[dataframe["selected_api_gptj"] == "MT"]

    gptj_bleu = None
    pure_gptj_bleu = None
    art_gptj_bleu = None
    pure_art_gptj_bleu = None
    chatgpt_bleu = None
    pure_chatgpt_bleu = None
    chatgpt_zeroshot_bleu = None
    gptj_zeroshot_bleu = None
    gptj_fewshot_bleu = None
    chatgpt_fewshot_bleu = None
    art_chatgpt_bleu = None
    pure_art_chatgpt_bleu = None
    

    if artdataframe is not None:
        art_gptj_bleu = artdataframe.apply(lambda x: calculate_bleu(x["answer"], x["gold_answer"]), axis=1).mean()
        pure_art_gptj_bleu = pure_art_gptj.apply(lambda x: calculate_bleu(x["answer"], x["gold_answer"]), axis=1).mean()
        art_chatgpt_bleu = artdataframe_chatgpt.apply(lambda x: calculate_bleu(x["answer"], x["gold_answer"]), axis=1).mean()
        pure_art_chatgpt_bleu = pure_art_chatgpt.apply(lambda x: calculate_bleu(x['answer'], x['gold_answer']), axis=1).mean()

    if "answer_gptj" in dataframe.columns:
        gptj_bleu = dataframe.apply(lambda x: calculate_bleu(x["answer_gptj"], x["gold_answer"]), axis=1).mean()
        pure_gptj_bleu = pure_gptj.apply(lambda x: calculate_bleu(x["answer_gptj"], x["gold_answer"]), axis=1).mean()

    if "answer_chatgpt" in dataframe.columns:
        chatgpt_bleu = dataframe.apply(lambda x: calculate_bleu(x["answer_chatgpt"], x["gold_answer"]), axis=1).mean()
        pure_chatgpt_bleu = pure_chatgpt.apply(lambda x: calculate_bleu(x["answer_chatgpt"], x["gold_answer"]), axis=1).mean()

    if "answer_chatgpt_zero_shot" in dataframe.columns:
        dataframe["answer_chatgpt_zero_shot"] = dataframe["answer_chatgpt_zero_shot"].replace({np.nan: None})
        chatgpt_zeroshot_bleu = dataframe.apply(lambda x: calculate_bleu(x["answer_chatgpt_zero_shot"], x["gold_answer"]), axis=1).mean()

    if "answer_gptj_zero_shot" in dataframe.columns:
        dataframe["answer_gptj_zero_shot"] = dataframe["answer_gptj_zero_shot"].replace({np.nan: None})
        gptj_zeroshot_bleu = dataframe.apply(lambda x: calculate_bleu(x["answer_gptj_zero_shot"], x["gold_answer"]), axis=1).mean()
    
    if "answer_gptj_few_shot" in dataframe.columns:
        dataframe["answer_gptj_few_shot"] = dataframe["answer_gptj_few_shot"].replace({np.nan: None})
        gptj_fewshot_bleu = dataframe.apply(lambda x: calculate_bleu(x["answer_gptj_few_shot"], x["gold_answer"]), axis=1).mean()

    if "answer_chatgpt_few_shot" in dataframe.columns:
        dataframe["answer_chatgpt_few_shot"] = dataframe["answer_chatgpt_few_shot"].replace({np.nan: None})
        chatgpt_fewshot_bleu = dataframe.apply(lambda x: calculate_bleu(x["answer_chatgpt_few_shot"], x["gold_answer"]), axis=1).mean()

    return {
        "select_accuracy": select_accuracy,
        "select_accuracy_medium": select_accuracy_medium,
        "select_accuracy_large": select_accuracy_large,
        "select_accuracy_xl": select_accuracy_xl,
        "select_accuracy_art": select_accuracy_art,
        "select_accuracy_10apis": select_accuracy_10apis,
        "gptj_bleu": gptj_bleu,
        "pure_gptj_bleu": pure_gptj_bleu,
        "art_gptj_bleu": art_gptj_bleu,
        "pure_art_gptj_bleu": pure_art_gptj_bleu,
        "chatgpt_bleu": chatgpt_bleu,
        "pure_chatgpt_bleu": pure_chatgpt_bleu,
        'art_chatgpt_bleu': art_chatgpt_bleu,
        'pure_art_chatgpt_bleu': pure_art_chatgpt_bleu,
        "chatgpt_zeroshot_bleu": chatgpt_zeroshot_bleu,
        "gptj_zeroshot_bleu": gptj_zeroshot_bleu,
        "gptj_fewshot_bleu": gptj_fewshot_bleu,
        "chatgpt_fewshot_bleu": chatgpt_fewshot_bleu
    }



def read_mlqa_dataset(dataframe):
    select_accuracy = dataframe.apply(lambda x: get_filtered_api(x) == "MultilingualQA", axis=1).sum() / len(dataframe) 
    
    pure_gptj = dataframe[dataframe["selected_api_gptj"] == "MultilingualQA"]

    gptj_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_gptj"]), axis=1).mean()
    pure_gptj_accuracy = pure_gptj.apply(lambda x: match_answer(x["gold_answer"], x["answer_gptj"]), axis=1).mean()

    return {
        "select_accuracy": select_accuracy,
        "gptj_accuracy": gptj_accuracy,
        "pure_gptj_accuracy": pure_gptj_accuracy
    }


def read_wiki_dataset(dataframe, artdataframe=None, artdataframe_chatgpt=None):
    select_accuracy = dataframe.apply(lambda x: get_filtered_api(x) == "WikiSearch", axis=1).sum() / len(dataframe)

    select_accuracy_medium = None
    select_accuracy_large = None
    select_accuracy_xl = None
    select_accuracy_art = None
    select_accuracy_10apis = None

    if "filtered_apis_medium" in dataframe.columns:
        select_accuracy_medium = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_medium") == "WikiSearch", axis=1).sum() / len(dataframe[dataframe["filtered_apis_medium"].notna()])

    if "filtered_apis_large" in dataframe.columns:
        select_accuracy_large = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_large") == "WikiSearch", axis=1).sum() / len(dataframe[dataframe["filtered_apis_large"].notna()])

    if "filtered_apis_xl" in dataframe.columns:
        select_accuracy_xl = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_xl") == "WikiSearch", axis=1).sum() / len(dataframe[dataframe["filtered_apis_xl"].notna()])

    if "filtered_10apis" in dataframe.columns:
        select_accuracy_10apis = dataframe.apply(lambda x: get_filtered_api(x, "filtered_10apis") == "WikiSearch", axis=1).sum() / len(dataframe[dataframe["filtered_10apis"].notna()])

    if artdataframe is not None:
        select_accuracy_art = artdataframe.iloc[0]["selection_ratio"]
        pure_art_gptj = artdataframe[artdataframe["task"] == "open_domain_qa"]
        pure_art_chatgpt = artdataframe_chatgpt[artdataframe_chatgpt['task'] == 'open_domain_qa']

    pure_chatgpt = dataframe[dataframe["selected_api_chatgpt"] == "WikiSearch"]
    pure_gptj = dataframe[dataframe["selected_api_gptj"] == "WikiSearch"]

    gptj_accuracy = None
    pure_gptj_accuracy = None
    chatgpt_accuracy = None
    art_gptj_accuracy = None
    pure_art_gptj_accuracy = None
    pure_chatgpt_accuracy = None
    chatgpt_zeroshot_accuracy = None
    gptj_fewshot_accuracy = None
    chatgpt_fewshot_accuracy = None
    gptj_zeroshot_accuracy = None
    art_chatgpt_accuracy = None
    pure_art_chatgpt_accuracy = None 

    if artdataframe is not None:
        art_gptj_accuracy = artdataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer"]), axis=1).mean()
        pure_art_gptj_accuracy = pure_art_gptj.apply(lambda x: match_answer(x["gold_answer"], x["answer"]), axis=1).mean()
        art_chatgpt_accuracy = artdataframe_chatgpt.apply(lambda x: match_answer(x["gold_answer"], x["answer"]), axis=1).mean()
        pure_art_chatgpt_accuracy = pure_art_chatgpt.apply(lambda x: match_answer(x['gold_answer'], x['answer']), axis=1).mean()

    if "answer_gptj" in dataframe.columns:
        gptj_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_gptj"]), axis=1).mean()
        pure_gptj_accuracy = pure_gptj.apply(lambda x: match_answer(x["gold_answer"], x["answer_gptj"]), axis=1).mean()
    
    if "answer_chatgpt" in dataframe.columns:
        chatgpt_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_chatgpt"]), axis=1).mean()
        pure_chatgpt_accuracy = pure_chatgpt.apply(lambda x: match_answer(x["gold_answer"], x["answer_chatgpt"]), axis=1).mean()

    if "answer_chatgpt_zero_shot" in dataframe.columns:
        chatgpt_zeroshot_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_chatgpt_zero_shot"]), axis=1).mean()

    if "answer_gptj_zero_shot" in dataframe.columns:
        gptj_zeroshot_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_gptj_zero_shot"]), axis=1).mean()

    if "answer_gptj_few_shot" in dataframe.columns:
        gptj_fewshot_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_gptj_few_shot"]), axis=1).mean()

    if "answer_chatgpt_few_shot" in dataframe.columns:
        chatgpt_fewshot_accuracy = dataframe.apply(lambda x: match_answer(x["gold_answer"], x["answer_chatgpt_few_shot"]), axis=1).mean()

    return {
        "select_accuracy": select_accuracy,
        "select_accuracy_medium": select_accuracy_medium,
        "select_accuracy_large": select_accuracy_large,
        "select_accuracy_xl": select_accuracy_xl,
        "select_accuracy_art": select_accuracy_art,
        "select_accuracy_10apis": select_accuracy_10apis,
        "gptj_accuracy": gptj_accuracy,
        "pure_gptj_accuracy": pure_gptj_accuracy,
        "art_gptj_accuracy": art_gptj_accuracy,
        "pure_art_gptj_accuracy": pure_art_gptj_accuracy,
        "chatgpt_accuracy": chatgpt_accuracy,
        "pure_chatgpt_accuracy": pure_chatgpt_accuracy,
        'art_chatgpt_accuracy': art_chatgpt_accuracy,
        'pure_art_chatgpt_accuracy': pure_art_chatgpt_accuracy,
        "chatgpt_zeroshot_accuracy": chatgpt_zeroshot_accuracy,
        "gptj_zeroshot_accuracy": gptj_zeroshot_accuracy,
        "gptj_fewshot_accuracy": gptj_fewshot_accuracy,
        "chatgpt_fewshot_accuracy": chatgpt_fewshot_accuracy
    }

def read_multiplechoice_dataset(dataframe, artdataframe=None, artdataframe_chatgpt=None):
    select_accuracy = dataframe.apply(lambda x: get_filtered_api(x) == "QA", axis=1).sum() / len(dataframe)

    select_accuracy_medium = None
    select_accuracy_large = None
    select_accuracy_xl = None
    select_accuracy_art = None
    select_accuracy_10apis = None

    if "filtered_apis_medium" in dataframe.columns:
        select_accuracy_medium = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_medium") == "QA", axis=1).sum() / len(dataframe[dataframe["filtered_apis_medium"].notna()])
    
    if "filtered_apis_large" in dataframe.columns:
        select_accuracy_large = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_large") == "QA", axis=1).sum() / len(dataframe[dataframe["filtered_apis_large"].notna()])

    if "filtered_apis_xl" in dataframe.columns:
        select_accuracy_xl = dataframe.apply(lambda x: get_filtered_api(x, "filtered_apis_xl") == "QA", axis=1).sum() / len(dataframe[dataframe["filtered_apis_xl"].notna()])

    if "filtered_10apis" in dataframe.columns:
        select_accuracy_10apis = dataframe.apply(lambda x: get_filtered_api(x, "filtered_10apis") == "QA", axis=1).sum() / len(dataframe[dataframe["filtered_10apis"].notna()])

    if artdataframe is not None:
        select_accuracy_art = artdataframe.iloc[0]["selection_ratio"]
        pure_art_gptj = artdataframe[artdataframe["task"] == "commonsense_qa"]
        pure_art_chatgpt = artdataframe_chatgpt[artdataframe_chatgpt['task'] == 'commonsense_qa']

    pure_chatgpt = dataframe[dataframe["selected_api_chatgpt"] == "QA"]
    pure_gptj = dataframe[dataframe["selected_api_gptj"] == "QA"]

    gptj_accuracy = None
    pure_gptj_accuracy = None
    chatgpt_accuracy = None
    art_gptj_accuracy = None
    pure_art_gptj_accuracy = None
    pure_chatgpt_accuracy = None
    chatgpt_zeroshot_accuracy = None
    gptj_fewshot_accuracy = None
    chatgpt_fewshot_accuracy = None
    gptj_zeroshot_accuracy = None
    art_chatgpt_accuracy = None
    pure_art_chatgpt_accuracy = None


    if artdataframe is not None:
        art_gptj_accuracy = artdataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer"]), axis=1).mean()
        pure_art_gptj_accuracy = pure_art_gptj.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer"]), axis=1).mean()
        art_chatgpt_accuracy = artdataframe_chatgpt.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer"]), axis=1).mean()
        pure_art_chatgpt_accuracy = pure_art_chatgpt.apply(lambda x: match_multiplechoice_answer(x['gold_answer'], x['answer']), axis=1).mean()

    if "answer_gptj" in dataframe.columns:
        gptj_accuracy = dataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_gptj"]), axis=1).mean()
        pure_gptj_accuracy = pure_gptj.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_gptj"]), axis=1).mean()

    if "answer_chatgpt" in dataframe.columns:
        chatgpt_accuracy = dataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_chatgpt"]), axis=1).mean()
        pure_chatgpt_accuracy = pure_chatgpt.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_chatgpt"]), axis=1).mean()

    if "answer_chatgpt_zero_shot" in dataframe.columns:
        chatgpt_zeroshot_accuracy = dataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_chatgpt_zero_shot"]), axis=1).mean()

    if "answer_gptj_zero_shot" in dataframe.columns:
        gptj_zeroshot_accuracy = dataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_gptj_zero_shot"]), axis=1).mean()

    if "answer_gptj_few_shot" in dataframe.columns:
        gptj_fewshot_accuracy = dataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_gptj_few_shot"]), axis=1).mean()
    
    if "answer_chatgpt_few_shot" in dataframe.columns:
        chatgpt_fewshot_accuracy = dataframe.apply(lambda x: match_multiplechoice_answer(x["gold_answer"], x["answer_chatgpt_few_shot"]), axis=1).mean()
    
    return {
        "select_accuracy": select_accuracy,
        "select_accuracy_medium": select_accuracy_medium,
        "select_accuracy_large": select_accuracy_large,
        "select_accuracy_xl": select_accuracy_xl,
        "select_accuracy_10apis": select_accuracy_10apis,
        "select_accuracy_art": select_accuracy_art,
        "gptj_accuracy": gptj_accuracy,
        "pure_gptj_accuracy": pure_gptj_accuracy,
        "art_gptj_accuracy": art_gptj_accuracy,
        "pure_art_gptj_accuracy": pure_art_gptj_accuracy,
        "chatgpt_accuracy": chatgpt_accuracy,
        "pure_chatgpt_accuracy": pure_chatgpt_accuracy,
        'art_chatgpt_accuracy': art_chatgpt_accuracy,
        'pure_art_chatgpt_accuracy': pure_art_chatgpt_accuracy,
        "chatgpt_zeroshot_accuracy": chatgpt_zeroshot_accuracy,
        "gptj_zeroshot_accuracy": gptj_zeroshot_accuracy,
        "gptj_fewshot_accuracy": gptj_fewshot_accuracy,
        "chatgpt_fewshot_accuracy": chatgpt_fewshot_accuracy
    }


def read_json(file, index_value="question_id", index_column=None):

    # import json
    # js = json.load(open(file))
    # for i in range(len(js)):
    #     if js[i]["filtered_apis"] == []:
    #         js[i]["filtered_apis"] = None

    # json.dump(js, open(file, 'w'), indent=4)
    df = pd.read_json(file)
    if index_column is not None:
        df[index_value] = index_column
    df.set_index(index_value, inplace=True)
    return df

# ASDIV 

d2 = read_json('../experiment/ASDiv/results/ASDiv_answers_t2.json')
d3 = read_json('../experiment/ASDiv/results/ASDiv_answers_t3.json')
d4 = read_json('../experiment/ASDiv/results/ASDiv_answers_t4.json')
d5 = read_json('../experiment/ASDiv/results/ASDiv_4apis_mediumfilter.json')
d6 = read_json('../experiment/ASDiv/results/ASDiv_4apis_largefilter.json')
d7 = read_json('../experiment/ASDiv/results/ASDiv_4apis_xlfilter.json')
d8 = read_json('../experiment/ASDiv/results/ASDiv_10apis_filter_075.json')


d2["filtered_apis_medium"] = d5["filtered_apis"]
d2["filtered_apis_large"] = d6["filtered_apis"]
d2["filtered_apis_xl"] = d7["filtered_apis"]
d2["filtered_10apis"] = d8["filtered_apis"]
d2["answer_chatgpt_zero_shot"] = d3["answer_chatgpt_zero_shot"].apply(process_potential_answers)
# d2["answer_gptj_zero_shot"] = d3["answer_gptj_zero_shot"].apply(process_potential_answers)
d2["answer_gptj_few_shot"] = d4["answer_gptj_few_shot"].apply(process_potential_answers)
d2["answer_chatgpt_few_shot"] = d4["answer_chatgpt_few_shot"].apply(process_potential_answers)
d2

art1 = read_json('../experiment/ASDiv/results/ASDiv_4apis_art_small_filter.json', index_value="id")
art2 = read_json('../experiment/ASDiv/results/ASDiv_4apis_art_small-small.json', index_value="id", index_column=art1.index)
art3 = read_json('../experiment/ASDiv/results/ASDiv_4apis_art_small-large.json', index_value="id")


data_asdiv = d2, art2, art3
read_math_dataset(d2, art2, art3)

 
# GSM8K

d1 = read_json("../experiment/GSM8K/result/GSM8K_4apis_t1.json")
d2 = read_json("../experiment/GSM8K/result/GSM8K_4apis_t2.json")
d3 = read_json("../experiment/GSM8K/result/GSM8K_4apis_mediumfilter.json")
d4 = read_json("../experiment/GSM8K/result/GSM8K_4apis_largefilter.json")
d5 = read_json("../experiment/GSM8K/result/GSM8K_4apis_xlfilter.json")
d6 = read_json("../experiment/GSM8K/result/GSM8K_10apis_filter_075.json")

d1["answer_chatgpt_zero_shot"] = d1["answer_chatgpt_zero_shot"].apply(process_potential_answers)
d1['answer_gptj_zero_shot'] = d1['answer_gptj_zero_shot'].apply(process_potential_answers)
d1["answer_chatgpt_few_shot"] = d2["answer_chatgpt_few_shot"].apply(process_potential_answers)
d1["answer_gptj_few_shot"] = d2["answer_gptj_few_shot"].apply(process_potential_answers)
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]

art1 = read_json('../experiment/GSM8K/result/GSM8K_4apis_art_small_filter.json', index_value="id")
art2 = read_json('../experiment/GSM8K/result/GSM8K_4apis_art_small-small.json', index_value="id", index_column=art1.index)
art3 = read_json('../experiment/GSM8K/result/GSM8K_4apis_art_small-large.json', index_value="id")

data_gsm8k = d1, art2, art3

# read_math_dataset(d1, art2, art3)

# SVAMP

d1 = read_json("../experiment/SVAMP/result/SVAMP_4apis_t1.json")
d2 = read_json("../experiment/SVAMP/result/SVAMP_4apis_t2.json")
d3 = read_json("../experiment/SVAMP/result/SVAMP_4apis_t3.json")
d4 = read_json("../experiment/SVAMP/result/SVAMP_4apis_mediumfilter.json")
d5 = read_json("../experiment/SVAMP/result/SVAMP_4apis_largefilter.json")
d6 = read_json("../experiment/SVAMP/result/SVAMP_4apis_xlfilter.json")
d7 = read_json("../experiment/SVAMP/result/SVAMP_10apis_filter_075.json")

d1.index = d2.index

d2["answer_chatgpt_zero_shot"] = d1["answer_chatgpt_zero_shot"].apply(process_potential_answers)
d2['answer_gptj_zero_shot'] = d1['answer_gptj_zero_shot'].apply(process_potential_answers)
d2['gold_answer'] = d2['gold_answer'].apply(str)
d2["answer_chatgpt_few_shot"] = d3["answer_chatgpt_few_shot"].apply(process_potential_answers)
d2["answer_gptj_few_shot"] = d3["answer_gptj_few_shot"].apply(process_potential_answers)
d2["filtered_apis_medium"] = d4["filtered_apis"]
d2["filtered_apis_large"] = d5["filtered_apis"]
d2["filtered_apis_xl"] = d6["filtered_apis"]
d2["filtered_10apis"] = d7["filtered_apis"]

art1 = read_json('../experiment/SVAMP/result/SVAMP_4apis_art_small_filter.json', index_value="id")
art2 = read_json('../experiment/SVAMP/result/SVAMP_4apis_art_small-small.json', index_value="id", index_column=art1.index)
art3 = read_json('../experiment/SVAMP/result/SVAMP_4apis_art_small-large.json', index_value="id")
art2["gold_answer"] = art2["gold_answer"].apply(str)
art3['gold_answer'] = art3["gold_answer"].apply(str)


data_svamp = d2, art2, art3


# read_math_dataset(d2, art2, art3)


# IWSLT cn2en

d1 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_t1.json')
d2 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_t3_gptjzeroshot.json')
d3 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_fewshot.json')
d4 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_4apis_mediumfilter.json')
d5 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_4apis_largefilter.json')
d6 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_4apis_xlfilter.json')
d7 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_10apis_filter_075.json')

d1["answer_gptj_zero_shot"] = d2["answer_gptj_zero_shot"]
d1["answer_chatgpt_few_shot"] = d3["answer_chatgpt_few_shot"]
d1["answer_gptj_few_shot"] = d3["answer_gptj_few_shot"]
d1["filtered_apis_medium"] = d4["filtered_apis"]
d1["filtered_apis_large"] = d5["filtered_apis"]
d1["filtered_apis_xl"] = d6["filtered_apis"]
d1["filtered_10apis"] = d7["filtered_apis"]

art1 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_4apis_art_small_filter.json', index_value="id")
art2 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_4apis_art_small-small.json', index_value="id", index_column=art1.index)
art3 = read_json('../experiment/IWSLT/cn/result/IWSLT_cn2en_4apis_art_small-large.json', index_value="id")


data_iwslt_cn2en = d1, art2, art3

# read_mt_dataset(d1, art2, art3)


# IWSLT ar2en

d1 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_t4.json')
d2 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_fewshot.json')
d3 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_4apis_mediumfilter.json')
d4 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_4apis_largefilter.json')
d5 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_4apis_xlfilter.json')
d6 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_10apis_filter_075.json')
d1["answer_chatgpt_few_shot"] = d2["answer_chatgpt_few_shot"]
d1["answer_gptj_few_shot"] = d2["answer_gptj_few_shot"]
d1["filtered_10apis"] = d6["filtered_apis"]

art2 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/IWSLT/ar/result/IWSLT_ar2en_4apis_art_small-large.json', index_value="id")
data_iwslt_ar2en = d1, art2, art3

# read_mt_dataset(d1, art2, art3)

# IWSLT de2en

d1 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_t5.json')
d2 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_fewshot.json')
d3 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_4apis_mediumfilter.json')
d4 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_4apis_largefilter.json')
d5 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_4apis_xlfilter.json')
d6 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_10apis_filter_075.json')

d1["answer_chatgpt_few_shot"] = d2["answer_chatgpt_few_shot"]
d1["answer_gptj_few_shot"] = d2["answer_gptj_few_shot"]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]

art1 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_4apis_art_small_filter.json', index_value="id")
art2 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_4apis_art_small-small.json', index_value="id", index_column=art1.index)
art3 = read_json('../experiment/IWSLT/de/result/IWSLT_de2en_4apis_art_small-large.json', index_value="id")


data_iwslt_de2en = d1, art2, art3

# IWSLT fr2en

d1 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_t6.json')
d2 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_fewshot.json')
d3 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_4apis_mediumfilter.json')
d4 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_4apis_largefilter.json')
d5 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_4apis_xlfilter.json')
d6 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_10apis_filter_075.json')

d1["answer_chatgpt_few_shot"] = d2["answer_chatgpt_few_shot"]
d1["answer_gptj_few_shot"] = d2["answer_gptj_few_shot"]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]


art1 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_4apis_art_small_filter.json', index_value="id")
art2 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_4apis_art_small-small.json', index_value="id", index_column=art1.index)
art3 = read_json('../experiment/IWSLT/fr/result/IWSLT_fr2en_4apis_art_small-large.json', index_value="id")

data_iwslt_fr2en = d1, art2, art3

# read_mt_dataset(d1, art2, art3)



# IWSLT ja2en

d1 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_t7.json')
d2 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_fewshot.json')
d3 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_4apis_mediumfilter.json')
d4 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_4apis_largefilter.json')
d5 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_4apis_xlfilter.json')
d6 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_10apis_filter_075.json')
d1["answer_chatgpt_few_shot"] = d2["answer_chatgpt_few_shot"]
d1["answer_gptj_few_shot"] = d2["answer_gptj_few_shot"]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]

art2 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/IWSLT/ja/result/IWSLT_ja2en_4apis_art_small-large.json', index_value="id")

data_iwslt_ja2en = d1, art2, art3

# read_mt_dataset(d1, art2, art3)

# IWSLT ko2en

d1 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_t8.json')
d2 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_fewshot.json')
d3 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_4apis_mediumfilter.json')
d4 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_4apis_largefilter.json')
d5 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_4apis_xlfilter.json')
d6 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_10apis_filter_075.json')
d1["answer_chatgpt_few_shot"] = d2["answer_chatgpt_few_shot"]
d1["answer_gptj_few_shot"] = d2["answer_gptj_few_shot"]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]

art2 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/IWSLT/ko/result/IWSLT_ko2en_4apis_art_small-large.json', index_value="id")


data_iwslt_ko2en = d1, art2, art3

# read_mt_dataset(d1, art2, art3)

# NQ-Open

d1 = read_json('../experiment/NQ-Open/result/NQ-Open_4apis_t1.json')
d2 = read_json('../experiment/NQ-Open/result/NQ-Open_fewshot.json')
d3 = read_json('../experiment/NQ-Open/result/NQ-Open_4apis_mediumfilter.json')
d4 = read_json('../experiment/NQ-Open/result/NQ-Open_4apis_largefilter.json')
d5 = read_json('../experiment/NQ-Open/result/NQ-Open_4apis_xlfilter.json')
d6 = read_json('../experiment/NQ-Open/result/NQ-Open_10apis_filter_075.json')

d1[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]] = d2[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]

art2 = read_json('../experiment/NQ-Open/result/NQ-Open_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/NQ-Open/result/NQ-Open_4apis_art_small-large.json', index_value="id")


data_nqopen = d1, art2, art3

# WebQS

d1 = read_json('../experiment/WebQS/result/WebQS_4apis_t1.json')
d2 = read_json('../experiment/WebQS/result/WebQS_fewshot.json')
d3 = read_json('../experiment/WebQS/result/WebQS_4apis_mediumfilter.json')
d4 = read_json('../experiment/WebQS/result/WebQS_4apis_largefilter.json')
d5 = read_json('../experiment/WebQS/result/WebQS_4apis_xlfilter.json')
d6 = read_json('../experiment/WebQS/result/WebQS_10apis_filter_075.json')

d1[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]] = d2[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d6 = d6[~d6.index.duplicated()]
d1["filtered_10apis"] = d6["filtered_apis"]


art2 = read_json('../experiment/WebQS/result/WebQS_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/WebQS/result/WebQS_4apis_art_small-large.json', index_value="id")


data_webqs = d1, art2, art3


# TriviaQA

d3 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_t1.json')
d2 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_t3.json')
d1 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_t4.json')
d4 = read_json('../experiment/TriviaQA/result/TriviaQA_fewshot.json')
d5 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_mediumfilter.json')
d6 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_largefilter.json')
d7 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_xlfilter.json')
d8 = read_json('../experiment/TriviaQA/result/TriviaQA_10apis_filter_075.json')


d1["answer_gptj_zero_shot"] = d2["answer_gptj_zero_shot"]
d1["answer_chatgpt_zero_shot"] = d3["answer_chatgpt_zero_shot"]
d1[['answer_gptj_few_shot', 'answer_chatgpt_few_shot']] = d4[['answer_gptj_few_shot', 'answer_chatgpt_few_shot']]
d1["filtered_apis_medium"] = d5["filtered_apis"]
d1["filtered_apis_large"] = d6["filtered_apis"]
d1["filtered_apis_xl"] = d7["filtered_apis"]
d1["filtered_10apis"] = d8["filtered_apis"]

art2 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/TriviaQA/result/TriviaQA_4apis_art_small-large.json', index_value="id")

data_triviaqa = d1, art2, art3

# read_wiki_dataset(d1, art2, art3)

# CommonsenseQA
tempd1 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_t2.json')

d1 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_t1.json')
d2 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_fewshot.json')
d3 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_mediumfilter.json')
d4 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_largefilter.json')
d5 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_xlfilter.json')
d6 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_10apis_filter_075.json')

d1[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]] = d2[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]

# d1[['answer_chatgpt', 'answer_gptj']] = tempd1[['answer_chatgpt', 'answer_gptj']]

art2 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/CommonsenseQA/result/CommonsenseQA_4apis_art_small-large.json', index_value="id")

data_commonsense_qa = d1, art2, art3

# COPA

tempd2 = read_json('../experiment/COPA/result/COPA_4apis_t2.json')

d1 = read_json('../experiment/COPA/result/COPA_4apis_t1.json')
d2 = read_json('../experiment/COPA/result/COPA_fewshot.json')
d3 = read_json('../experiment/COPA/result/COPA_4apis_mediumfilter.json')
d4 = read_json('../experiment/COPA/result/COPA_4apis_largefilter.json')
d5 = read_json('../experiment/COPA/result/COPA_4apis_xlfilter.json')
d6 = read_json('../experiment/COPA/result/COPA_10apis_filter_075.json')

d1[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]] = d2[["answer_gptj_few_shot", "answer_chatgpt_few_shot"]]
d1["filtered_apis_medium"] = d3["filtered_apis"]
d1["filtered_apis_large"] = d4["filtered_apis"]
d1["filtered_apis_xl"] = d5["filtered_apis"]
d1["filtered_10apis"] = d6["filtered_apis"]


art2 = read_json('../experiment/COPA/result/COPA_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/COPA/result/COPA_4apis_art_small-large.json', index_value="id")


data_copa = d1, art2, art3

# read_multiplechoice_dataset(d1, art2, art3)

# SocialIQA

d1 = read_json('../experiment/SocialIQA/result/SocialIQA_4apis_t1.json')
d2 = read_json('../experiment/SocialIQA/result/SocialIQA_4apis_mediumfilter.json')
d3 = read_json('../experiment/SocialIQA/result/SocialIQA_4apis_largefilter.json')
d4 = read_json('../experiment/SocialIQA/result/SocialIQA_10apis_filter_075.json')
tempd3 = read_json('../experiment/SocialIQA/result/SocialIQA_4apis_t2.json')

d1["filtered_apis_medium"] = d2["filtered_apis"]
d1["filtered_apis_large"] = d3["filtered_apis"]
d1["filtered_10apis"] = d4["filtered_apis"]

art2 = read_json('../experiment/SocialIQA/result/SocialIQA_4apis_art_small-small.json', index_value="id")
art3 = read_json('../experiment/SocialIQA/result/SocialIQA_4apis_art_small-large.json', index_value="id")


data_social_iqa = d1, art2, art3
# read_multiplechoice_dataset(d1, art2, art3)


d1 = read_json('../experiment/MLQA/es/result/MLQA_es2en_5apis.json')
read_mlqa_dataset(d1)



result = pd.DataFrame(columns=["API Ratio", "Medium Ratio", "Large Ratio", "XL Ratio", "GPT2 Medium 10APIs Ratio", "ART Small Ratio", "GPT-J", "Pure GPT-J", "ART GPT-J", "Pure ART GPT-J", "GPTJ Zero-shot", "GPTJ Few-shot", "ChatGPT", "Pure ChatGPT", "ART ChatGPT", "Pure ART ChatGPT", "ChatGPT Zero-shot", "ChatGPT Few-shot"])

math_datasets = {
    "ASDiv": data_asdiv, 
    "GSM8K": data_gsm8k, 
    "SVAMP": data_svamp
}


for key, data in math_datasets.items():
    if type(data) == tuple:
        d = read_math_dataset(data[0], data[1], data[2])
    else:
        d = read_math_dataset(data)
    result.loc[key] = [d["select_accuracy"],  d["select_accuracy_medium"], d["select_accuracy_large"], d["select_accuracy_xl"], d["select_accuracy_10apis"], d["select_accuracy_art"], d["gptj_accuracy"], d["pure_gptj_accuracy"], d["art_gptj_accuracy"], d["pure_art_gptj_accuracy"], d["gptj_zeroshot_accuracy"], d["gptj_fewshot_accuracy"], d["chatgpt_accuracy"], d["pure_chatgpt_accuracy"], d['art_chatgpt_accuracy'], d['pure_art_chatgpt_accuracy'], d["chatgpt_zeroshot_accuracy"], d["chatgpt_fewshot_accuracy"]]

mt_datasets = {
    "IWSLT_cn2en": data_iwslt_cn2en,
    "IWSLT_ar2en": data_iwslt_ar2en,
    "IWSLT_de2en": data_iwslt_de2en,
    "IWSLT_fr2en": data_iwslt_fr2en,
    "IWSLT_ja2en": data_iwslt_ja2en,
    "IWSLT_ko2en": data_iwslt_ko2en
}

for key, data in mt_datasets.items():
    if type(data) == tuple:
        d = read_mt_dataset(data[0], data[1], data[2])
    else:
        d = read_mt_dataset(data)

    result.loc[key] = [d["select_accuracy"], d["select_accuracy_medium"], d["select_accuracy_large"], d["select_accuracy_xl"], d["select_accuracy_10apis"], d["select_accuracy_art"], d["gptj_bleu"], d["pure_gptj_bleu"], d["art_gptj_bleu"], d["pure_art_gptj_bleu"], d["gptj_zeroshot_bleu"], d["gptj_fewshot_bleu"], d["chatgpt_bleu"], d["pure_chatgpt_bleu"], d['art_chatgpt_bleu'], d['pure_art_chatgpt_bleu'], d["chatgpt_zeroshot_bleu"], d["chatgpt_fewshot_bleu"]]

wiki_datasets = {
    "NQ-Open": data_nqopen,
    "WebQS": data_webqs,
    "TriviaQA": data_triviaqa
}

for key, data in wiki_datasets.items():
    if type(data) == tuple:
        d = read_wiki_dataset(data[0], data[1], data[2])
    else:
        d = read_wiki_dataset(data)

    result.loc[key] = [d["select_accuracy"], d["select_accuracy_medium"], d["select_accuracy_large"], d["select_accuracy_xl"], d["select_accuracy_10apis"], d["select_accuracy_art"], d["gptj_accuracy"], d["pure_gptj_accuracy"], d["art_gptj_accuracy"], d["pure_art_gptj_accuracy"], d["gptj_zeroshot_accuracy"], d["gptj_fewshot_accuracy"], d["chatgpt_accuracy"], d["pure_chatgpt_accuracy"], d['art_chatgpt_accuracy'], d['pure_art_chatgpt_accuracy'], d["chatgpt_zeroshot_accuracy"], d["chatgpt_fewshot_accuracy"]]

qa_datasets = {
    "CommonsenseQA": data_commonsense_qa,
    "newPromptcommonsenseQA": tempd1,
    "COPA": data_copa,
    "newPeomptCOPA": tempd2,
    "SocialIQA": data_social_iqa,
    "newPromptSocialIQA": tempd3
}

for key, data in qa_datasets.items():
    if type(data) == tuple:
        d = read_multiplechoice_dataset(data[0], data[1], data[2])
    else:
        d = read_multiplechoice_dataset(data)
    result.loc[key] = [d["select_accuracy"], d["select_accuracy_medium"], d["select_accuracy_large"], d["select_accuracy_xl"], d["select_accuracy_10apis"], d["select_accuracy_art"], d["gptj_accuracy"], d["pure_gptj_accuracy"], d["art_gptj_accuracy"], d["pure_art_gptj_accuracy"], d["gptj_zeroshot_accuracy"], d["gptj_fewshot_accuracy"], d["chatgpt_accuracy"], d["pure_chatgpt_accuracy"], d['art_chatgpt_accuracy'], d['pure_art_chatgpt_accuracy'], d["chatgpt_zeroshot_accuracy"], d["chatgpt_fewshot_accuracy"]]
         

result.to_csv("./result/result.csv")


# # confusion matrix
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # get key list of each dictionary in data_commonsense_qa[0]['filtered_apis']

# # create an empty pandas series

# y_pred = pd.Series()
# y_true = pd.Series()


# def get_y_pred_true(data, gold_tool):
#     global y_pred
#     global y_true
#     for key, data in data.items():
#         if type(data) == tuple:
#             current_data = data[0]
#         else:
#             current_data = data
#         y_pred = y_pred.append(current_data['filtered_apis'].apply(lambda x: list(x.keys())[0]),ignore_index = True)
#         y_true = y_true.append(pd.Series([gold_tool]*len(current_data['filtered_apis'])),ignore_index = True)
    
#     return y_pred, y_true

# qa_datasets = {
#     "CommonsenseQA": data_commonsense_qa,
#     "COPA": data_copa,
#     "SocialIQA": data_social_iqa,
# }

# # to list
# y_pred, y_true = get_y_pred_true(math_datasets, "Calculator")
# y_pred, y_true = get_y_pred_true(mt_datasets, 'MT')
# y_pred, y_true = get_y_pred_true(wiki_datasets, "WikiSearch")
# y_pred, y_true = get_y_pred_true(qa_datasets, "QA")

# classNames = ['Calculator', 'MT', 'WikiSearch', 'QA'] 

# cm = confusion_matrix(y_true, y_pred, labels=classNames)

# df_cm = pd.DataFrame(cm, columns=np.unique(y_true), index = np.unique(y_true))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'


# f, ax = plt.subplots(figsize=(8, 8))
# cmap = sns.cubehelix_palette(light=1, as_cmap=True)

# sns.heatmap(df_cm, cbar=True, annot=True, cmap=cmap, square=True, fmt='.0f',
#             annot_kws={'size': 10})
# plt.savefig('confusion_matrix.png',dpi=600)
