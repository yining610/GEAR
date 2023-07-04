from tool_ground import APIFilter
from tool_execute import LMAPIExecutor, OpenAIAPIExecutor, OpenAI_Zero_Executor
from read_dataset import *

# from prompt import mt_prompt, wiki_prompt, qa_prompt, calculator_prompt, zero_shot_prompt, timezoneconverter_prompt
# from prompt import mt_description, wiki_description, qa_description, calculator_description, context_qa_description, multilingual_description, timezoneconverter_description
from prompt import *
from fewshot_prompt import mt_prompt as mt_prompt_few_shot, wiki_prompt as wiki_prompt_few_shot, qa_prompt as qa_prompt_few_shot, calculator_prompt as calculator_prompt_few_shot
from api import *
from pattern import *
from utils import get_args

import json
import os
import time

def main(): 
    # apis
    calculator_api = CalculatorAPI("Calculator", calculator_description, calculator_prompt, pattern=NUMBER_PATTERN) # the output pattern is always number
    wiki_api = WikiSearchAPI("WikiSearch", wiki_description, wiki_prompt)

    qa_api = QAAPI("QA", qa_description, qa_prompt) 
    # same description and prompt as QA, but different model used behind. 
    # context_qa_api = ContextQAAPI("ContextQA", context_qa_description)
    
    mt_api = MTAPI("MT", mt_description, mt_prompt) # don't specify the output pattern for MT, becuse the output language is various
    multilingual_qa_api = MultiLingualAPI("MultilingualQA",  multilingual_description)

    timezone_api = TimeZoneAPI("TimezoneConverter", timezoneconverter_description, timezoneconverter_prompt, pattern=TIME_PATTERN)

    # tts_api = TTS("TTS", tts_description, tts_prompt)
    log_api = Log("Log", log_description, log_prompt, pattern=NUMBER_PATTERN)
    pow_api = Pow("Pow", pow_description, pow_prompt, pattern=NUMBER_PATTERN)

    robot_api = RobotMove("RobotMove", robot_description, robot_prompt, pattern=MOVE_PATTERN)
    sleep_api = Sleep("Sleep", sleep_description, sleep_prompt, pattern=SLEEP_PATTERN)

    
    # define APIs list
    # apis = [calculator_api, timezone_api, qa_api, mt_api]
    apis = []
    if "calculator" in args.tool:
        apis.append(calculator_api)
    if "wiki" in args.tool:
        apis.append(wiki_api)
    if "qa" in args.tool:
        apis.append(qa_api)
    if "mt" in args.tool:
        apis.append(mt_api)
    if "multilingualqa" in args.tool:
        apis.append(multilingual_qa_api)
    if "timezone" in args.tool:
        apis.append(timezone_api)
    if "sleep" in args.tool:
        apis.append(sleep_api)
    if "log" in args.tool:
        apis.append(log_api)
    if "exp" in args.tool:
        apis.append(pow_api)
    if "robot" in args.tool:
        apis.append(robot_api)

    name_2_api = {api.name: api for api in apis}

    start_time = time.time()
    API_filter = APIFilter(args, apis)
    if args.verbose:
        print("--- Load LM1 %s seconds ---" % (time.time() - start_time))

    if ("gptj" in args.experiment) or ("gptj_zero_shot" in args.experiment) or ("gptj_few_shot" in args.experiment):
        start_time = time.time()
        LMAPI_executor = LMAPIExecutor(args)
        if args.verbose:
            print("--- Load LM2 %s seconds ---" % (time.time() - start_time))
    if "openai" in args.experiment:
        OpenAI_executor = OpenAIAPIExecutor(args)
    if ("openai_zero_shot" in args.experiment) or ("openai_few_shot" in args.experiment):
        OpenAI_zeroshot_executor = OpenAI_Zero_Executor(args, zero_shot_prompt)

    file = args.dataset
    if args.check_point is not None:
        if os.path.exists(args.check_point):
            with open(args.check_point, "r") as f:
                LM1_output = json.load(f)
        else:
            LM1_output = {}
        if file not in LM1_output:
            LM1_output[file] = {}

    result = []

    # read dataset
    if "commonsenseqa" in file.lower():
        inputs, gold_answers, question_ids = read_commonsenseQA(file)
    elif "copa" in file.lower():
        inputs, gold_answers, question_ids = read_copa(file)
    elif "triviaqa" in file.lower():
        inputs, gold_answers, question_ids = read_TriviaQA(file)
    elif "social" in file.lower():
        inputs, gold_answers, question_ids = read_socialiqa(file)
    elif "asdiv" in file.lower():
        inputs, gold_answers, question_ids = read_ASDiv(file)
    elif "iwslt" in file.lower():
        inputs, gold_answers, question_ids = read_IWSLT(file)
    elif "webqs" in file.lower():
        inputs, gold_answers, question_ids = read_webqs(file)
    elif "gsm8k" in file.lower():
        inputs, gold_answers, question_ids = read_GSM8K(file)
    elif "svamp" in file.lower():
        inputs, gold_answers, question_ids = read_SVAMP(file)
    elif "nq-open" in file.lower():
        inputs, gold_answers, question_ids = read_nq_open(file)
    elif "timezone" in file.lower():
        inputs, gold_answers, question_ids = read_timezone(file)
    elif "mlqa" in file.lower():
        inputs, gold_answers, question_ids = read_MLQA(file)
    elif "t-rex" in file.lower():
        inputs, gold_answers, question_ids = read_trex(file)

    # GPTJ zero shot generation
    if "gptj_zero_shot" in args.experiment:
        if args.early_stop != 0 and args.early_stop < len(inputs):
            gptj_zero_shot_outputs = LMAPI_executor.zero_shot_batch(inputs[:args.early_stop], batch_size=4)
        else:
            gptj_zero_shot_outputs = LMAPI_executor.zero_shot_batch(inputs, batch_size=4)

    if "gptj_few_shot" in args.experiment:
        if args.prompt == "mt":
            few_shot_inputs = [mt_prompt_few_shot.format(input=input) for input in inputs]
        elif args.prompt == "wiki":
            few_shot_inputs = [wiki_prompt_few_shot.format(input=input) for input in inputs]
        elif args.prompt == "qa":
            few_shot_inputs = [qa_prompt_few_shot.format(input=input) for input in inputs]
        elif args.prompt == "calculator":
            few_shot_inputs = [calculator_prompt_few_shot.format(input=input) for input in inputs]
        else:
            few_shot_inputs = inputs.copy()

        if args.early_stop != 0 and args.early_stop < len(inputs):
            gptj_few_shot_outputs = LMAPI_executor.zero_shot_batch(few_shot_inputs[:args.early_stop], batch_size=4)
        else:
            gptj_few_shot_outputs = LMAPI_executor.zero_shot_batch(few_shot_inputs, batch_size=4)

    for idx, (input, gold_answer, question_id) in enumerate(zip(inputs, gold_answers, question_ids)):
        
        if idx == args.early_stop and args.early_stop != 0:
            break

        start_time = time.time()
        if args.verbose:
            print(f"***********************DATA {idx}***********************")
        
        question_id = str(question_id)
        if args.check_point is not None:
            if question_id in LM1_output[file]:
                # read from checkpoint
                filtered_apis_names = LM1_output[file][question_id]["filtered_apis"]
                filtered_apis = [name_2_api[name] for name in filtered_apis_names]
                filtered_apis_with_scores = LM1_output[file][question_id]["filtered_apis_with_scores"]
            else:
                filtered_apis, filtered_apis_with_scores = API_filter.filter(input)
                LM1_output[file][question_id] = {"filtered_apis": [api.name for api in filtered_apis], "filtered_apis_with_scores": filtered_apis_with_scores}
        else:
            try:
                filtered_apis, filtered_apis_with_scores = API_filter.filter(input)
            except:
                filtered_apis, filtered_apis_with_scores = None, None

        if args.verbose:
            print("--- Filter %s seconds ---" % (time.time() - start_time))

        if args.check_point is not None:
            # save checkpoint
            with open(args.check_point, "w") as f:
                json.dump(LM1_output, f, indent=4)

        current_result = {}
        current_result["question_id"] = question_id
        current_result["input"] = input
        current_result["gold_answer"] = gold_answer
        current_result["filtered_apis"] = filtered_apis_with_scores

        start_time = time.time()
        if "ground" in args.experiment:
            pass
        if "gptj" in args.experiment:
            selected_api_1, answer_1 = LMAPI_executor.execute(input, filtered_apis)
            current_result["answer_gptj"] = answer_1
            current_result['selected_api_gptj'] = selected_api_1
        if "openai" in args.experiment:
            selected_api_2, answer_2 = OpenAI_executor.execute(input, filtered_apis)
            current_result["answer_openai"] = answer_2
            current_result['selected_api_openai'] = selected_api_2
        if "openai_zero_shot" in args.experiment:
            answer_3 = OpenAI_zeroshot_executor.execute(input)
            current_result["answer_openai_zero_shot"] = answer_3
        if "gptj_zero_shot" in args.experiment:
            if args.verbose:
                print("***********************GPTJ Zero-Shot Result***********************")
                print(gptj_zero_shot_outputs[idx])
                print("***********************GPTJ Zero-Shot End***********************")
            current_result["answer_gptj_zero_shot"] = gptj_zero_shot_outputs[idx]
        if "gptj_few_shot" in args.experiment:
            if args.verbose:
                print("***********************GPTJ Few-Shot Result***********************")
                print(gptj_few_shot_outputs[idx])
                print("***********************GPTJ Few-Shot End***********************")
            current_result["answer_gptj_few_shot"] = gptj_few_shot_outputs[idx]
        if "openai_few_shot" in args.experiment:
            if args.prompt == "mt":
                selected_prompt = mt_prompt_few_shot
            elif args.prompt == "wiki":
                selected_prompt = wiki_prompt_few_shot
            elif args.prompt == "qa":
                selected_prompt = qa_prompt_few_shot
            elif args.prompt == "calculator":
                selected_prompt = calculator_prompt_few_shot

            answer_4 = OpenAI_zeroshot_executor.execute(input, selected_prompt)
            current_result["answer_openai_few_shot"] = answer_4
            
        if args.verbose: 
            print("--- Execute %s seconds ---" % (time.time() - start_time))

        result.append(current_result)


        with open(args.output, "w") as f:
            json.dump(result, f, indent=4)
            
if __name__ == "__main__":

    args = get_args()

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))