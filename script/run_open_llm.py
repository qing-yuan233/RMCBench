from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
import time
from datetime import datetime
import json
import re
import os
import argparse

model_info_dict = [
    {
        "mid":0,
        "dir_name":"llama-2-7b-chat-hf",
        "LLM_name":"Llama"
    },
    {
        "mid":1,
        "dir_name":"llama-2-13b-chat-hf",
        "LLM_name":"Llama"
    },
    {
        "mid":2,
        "dir_name":"tulu-2-13b",
        "LLM_name":"Tulu"
    },
    {
        "mid":3,
        "dir_name":"zephyr-7b-beta",
        "LLM_name":"Zephyr"
    },
    {
        "mid":4,
        "dir_name":"mpt-7b-chat",
        "LLM_name":"Mpt"
    },
    {
        "mid":5,
        "dir_name":"vicuna-7b-v1.3",
        "LLM_name":"Vicuna"
    },
    {
        "mid":6,
        "dir_name":"CodeLlama-13b-Instruct-hf",
        "LLM_name":"CodeLlama"
    },
    {
        "mid":7,
        "dir_name":"deepseek-coder-7b-instruct-v1.5",
        "LLM_name":"Deepseek"
    },
    {
        "mid":8,
        "dir_name":"Meta-Llama-3-8B-Instruct",
        "LLM_name":"Llama"
    }
]

model_name = "" 
pipeline = "" 
tokenizer = "" 

def format_prompt(user_message):
    global model_name
    if model_name in ["llama-2-7b-chat-hf","llama-2-13b-chat-hf","CodeLlama-13b-Instruct-hf"]:
        new_prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

{ user_message } [/INST]
"""
        return new_prompt

    if model_name == "tulu-2-13b":
        new_prompt = f"""<|user|>
{ user_message }
<|assistant|>
"""
        return  new_prompt

    if model_name == "zephyr-7b-beta":
        new_prompt = f"""<|system|>
you are a helpful assistant.</s>
<|user|>
{ user_message }</s>
<|assistant|>
"""
        return new_prompt
    
    if model_name == "mpt-7b-chat":
        new_prompt = f"""<|im_start|>user
{ user_message }<|im_end|>
<|im_start|>assistant
"""
        return new_prompt

    if model_name == "vicuna-7b-v1.3":
        new_prompt = f"""USER:
{ user_message }
ASSISTANT:
"""
        return  new_prompt

    if model_name == "deepseek-coder-7b-instruct-v1.5":
        new_prompt = f"""You are an AI programming assistant.
### Instruction:
{ user_message }
### Response:
"""
        return  new_prompt
        
    return user_message

    
def get_response(prompt):

    global pipeline,tokenizer

    user_message = prompt
    prompt = format_prompt(prompt)

    
    terminators = tokenizer.eos_token_id
    
    sequences = pipeline(
        prompt, 
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature = 0.75,
        num_return_sequences=1,
        max_new_tokens=4096, 
    )

    for seq in sequences:
      
        context = seq['generated_text'] 

        only_response = context.split(prompt, 1)[-1]
        return only_response

def change_LLM_name_in_prompt(old_prompt,LLM_name):
    if LLM_name == "chatgpt" :
        return old_prompt
    
    result_str = re.sub('chatgpt', LLM_name, old_prompt, flags=re.IGNORECASE)
    return result_str

def main():

    global model_info_dict,pipeline,tokenizer,model_name
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--run_model_index',type=int,required=True, help='index')
    parser.add_argument('--run_rounds', type=int,nargs='+',required=True, help='round')

    args = parser.parse_args()

    run_model_index = args.run_model_index
    run_rounds = args.run_rounds

    for run_round in run_rounds:

        file_path = f'' 
        model_info = model_info_dict[run_model_index]
        model_path = "" 
        output_file_dir = '' 
        LLM_name = model_info["LLM_name"] 

        model_name = model_path.split('/')[-1] 

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        )

        start_time = time.perf_counter()
        
        prompt_df = pd.read_excel(file_path)

        prompt_df['response'] = ''

        num_rows = len(prompt_df)
        print("total row:", num_rows)

        index = 0  
        same_error_times=0 
        error_id_list = [] 

        while index < num_rows:

            print(f"now is processing the index：{index}  data=============================")
            
            row = prompt_df.iloc[index]
            print("pid:",row['pid'])

            prompt = row['prompt']
            if row['category']=="text-to-code" and row["level"]==3:
                prompt = change_LLM_name_in_prompt(prompt,LLM_name)
                prompt_df.at[index, 'prompt'] = prompt 


            print("\nprompt:")
            print(prompt)

            response = get_response(prompt)

            print("\nLLM:")
            print(response)
            print("\n====================================================================================\n")

            response_line = response.count('\n') + 1
            if response_line >=500: 
                response  = "Long text exception output.Prevent memory overflow and replace it."
                print(response)

            prompt_df.at[index, 'response'] = str(response)

            index += 1

        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        output_file = output_file_dir + f"res_{model_name}_{timestamp}_round{run_round}_done.xlsx"
        print("out",output_file)

        json_name = output_file_dir + f"res_{model_name}_{timestamp}_round{run_round}_done.json"
        prompt_df.to_json(json_name,orient='records')
        prompt_df.to_excel(output_file, index=False)

        end_time = time.perf_counter()
        e2e_inference_time = (end_time-start_time)
        print(f"The time consuming is {e2e_inference_time} s")

        run_info = {
            "model_name":model_name,
            "run_time":timestamp,
            "time consuming":f"{e2e_inference_time} s",
            "error_id_list":error_id_list
        }

        with open(output_file_dir + f"run_info_{model_name}_{timestamp}_round{run_round}_done.json", 'w') as f:
            json.dump(run_info, f)

if __name__ == "__main__":
    main()

