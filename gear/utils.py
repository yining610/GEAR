import re
import argparse

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


def _extract_api_request_content(text: str, api_name:str) -> list:
    """Extract the content of an API request from a given text."""
    try:
        left_bracket = text.split(f"{api_name}(")[1]
        right_bracket_ind = left_bracket.rfind(")")
        inside = left_bracket[:right_bracket_ind]
    except Exception as e:
        return None

    request_args = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', inside)
    request_args = [arg.strip() for arg in request_args]
    request_args = [arg.replace('"', '') for arg in request_args]
    return request_args

# def _extract_api_request_content(text: str, api_name: str) -> str:
#     """Extract the content of an API request from a given text."""
#     if api_name != "MT":
#         start_tag = f"{api_name}("
#         end_tag = ")"
#         dest = None
#     else:
#         start_tag = f"{api_name}(\""
#         end_tag = "\", \""
#         dest_start_tag = f", \""
#         dest_end_tag = "\")"
#         dest_start_idx = text.find(dest_start_tag)
#         if dest_start_idx == -1:
#             dest = None
#         else:
#             dest_start_idx += len(dest_start_tag)
#             dest_end_idx = text.find(dest_end_tag, dest_start_idx)
#             if dest_end_idx == -1:
#                 dest = None
#             else:
#                 dest = text[dest_start_idx:dest_end_idx]
#     start_idx = text.find(start_tag)
#     if start_idx == -1:
#         return None, dest
#     start_idx += len(start_tag)
#     # find in reverse order
#     end_idx = text.rfind(end_tag, start_idx)
#     if end_idx == -1:
#         return None, dest
    
#     return text[start_idx:end_idx], dest


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--slm1", help="small language model name or path",
                        type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--slm2", help="small language model 2 name or path",
                        type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("-v", "--verbose", help="verbose",
                        action="store_true")
    parser.add_argument("-M", "--max_tokens", help="max tokens",
                         type=int, default=512)
    parser.add_argument("-t", "--top_k", help="Return top_k APIs",
                        type=int, default=1)
    parser.add_argument("--llm", help="large language model name or path",
                        type=str, default="EleutherAI/gpt-j-6B")
    parser.add_argument("-e", "--early_stop", help="early stop the program", 
                        type=int, default = 0)
    parser.add_argument("-c", "--check_point", help="file path of check point file", 
                        type=str, default=None)
    parser.add_argument("-d", "--dataset", help="dataset path",
                        type=str)
    parser.add_argument("-o", "--output", help="output path",
                        type=str, default=None)
    parser.add_argument('--experiment', choices=['gptj', 'gptj_zero_shot', 'openai', "gptj_few_shot", "openai_zero_shot", "openai_few_shot", "ground"], nargs="+")
    parser.add_argument('--openai_model', choices=['chatgpt', 'gpt3'], default="gpt3")
    parser.add_argument("--tool", choices=["calculator", "wiki", "qa", "mt", "multilingualqa", "timezone", "sleep", "log", "exp", "robot"], nargs="+")
    parser.add_argument('--prompt', choices=['mt', 'wiki', 'qa', 'calculator'], default="mt")
    parser.add_argument('--fdevice', type=str, default="cuda:0")
    parser.add_argument('--ALPHA', type=float, default=0.75)
    
    args = parser.parse_args()
    return args


