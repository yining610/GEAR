import pandas as pd
import numpy as np

def read_logs(input_files, output_file):
    result = {"Calculator": {"semantic": [], "pattern": [], "final": []}, 
          "WikiSearch": {"semantic": [], "pattern": [], "final": []}, 
          "QA": {"semantic": [], "pattern": [], "final": []}, 
          "MT": {"semantic": [], "pattern": [], "final": []}}

    semantic_score_list = [f"{tool} \t\t Semantic Score: " for tool in result.keys()]
    pattern_score_list = [f"{tool} pattern entailment score: " for tool in result.keys()]
    final_score_list = [f"{tool} final entailment score: " for tool in result.keys()]

    for input_file in input_files:
        with open(input_file) as f:
            lines = f.readlines()

    # time_used = 0
    # time_count = 0

        for line in lines:
            # skip empty lines
            if len(line.strip()) == 0:
                continue
            
            # if "--- Filter " in line:
                # time_used += float(line.split("--- Filter ")[-1].split(" seconds")[0])
                # time_count += 1

            for idx, tool in enumerate(result.keys()):    
                if semantic_score_list[idx] in line:
                    semantic_score_tool = float(line.split(semantic_score_list[idx])[-1].strip())
                    result[tool]["semantic"].append(semantic_score_tool)
                if pattern_score_list[idx] in line:
                    pattern_score_tool = float(line.split(pattern_score_list[idx])[-1].strip())
                    result[tool]["pattern"].append(pattern_score_tool)
                if final_score_list[idx] in line:
                    final_score_tool = float(line.split(final_score_list[idx])[-1].strip())
                    result[tool]["final"].append(final_score_tool)
        
    # print("avg filter time: ", time_used / time_count)
    # sanity check
    for tool in result.keys():
        assert len(result[tool]["semantic"]) == len(result[tool]["pattern"]) == len(result[tool]["final"])
        assert np.sum(0.75 * np.array(result[tool]["semantic"]) + 0.25 * np.array(result[tool]["pattern"])  - np.array(result[tool]["final"]))

    # find average semantic and pattern score for each tool
    for tool in result.keys():
        result[tool]["semantic"] = sum(result[tool]["semantic"]) / len(result[tool]["semantic"])
        result[tool]["pattern"] = sum(result[tool]["pattern"]) / len(result[tool]["pattern"])
        result[tool]["final"] = sum(result[tool]["final"]) / len(result[tool]["final"])

    # convert result to dataframe
    result_df = pd.DataFrame(result).T
    result_df = result_df[["semantic", "pattern", "final"]]
    result_df.to_csv(output_file)

math_files = [
    "../experiment/ASDiv/ASDiv_4apis_t2.txt",
    "../experiment/GSM8K/GSM8K_4apis_t1.txt",
    "../experiment/SVAMP/SVAMP_4apis_without_zeroshot_t2.txt",
]

mt_files = [
    "../experiment/IWSLT/IWSLT_4apis_cn2en_t1.txt",
    "../experiment/IWSLT/IWSLT_4apis_ar2en_t4.txt",
    "../experiment/IWSLT/IWSLT_4apis_de2en_t5.txt",
    "../experiment/IWSLT/IWSLT_4apis_fr2en_t6.txt",
    "../experiment/IWSLT/IWSLT_4apis_ja2en_t7.txt",
    "../experiment/IWSLT/IWSLT_4apis_ko2en_t8.txt",
]

wiki_files = [
    "../experiment/NQ-Open/NQ-Open_4apis_t1.txt",
    "../experiment/WebQS/WebQS_4apis_t1.txt",
    "../experiment/TriviaQA/TriviaQA_4apis_t4.txt",
]

qa_files = [
    "../experiment/CommonsenseQA/CommonsenseQA_4apis_t1.txt",
    "../experiment/COPA/COPA_4apis_t1.txt",
    "../experiment/SocialIQA/SocialIQA_4apis_t1.txt",
]

read_logs(math_files, "./result/math_4apis.csv")
read_logs(mt_files, "./result/mt_4apis.csv")
read_logs(wiki_files, "./result/wiki_4apis.csv")
read_logs(qa_files, "./result/qa_4apis.csv")
