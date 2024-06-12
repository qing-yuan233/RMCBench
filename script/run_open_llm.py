from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# from check_refuse import refuse_generate_malicious_code
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

model_name = "" # Model name (with parameters)
pipeline = ""
tokenizer = "" 
# Format according to different LLM prompt chat templates
def format_prompt(user_message):
    global model_name
    # llama2 series
    if model_name in ["llama-2-7b-chat-hf","llama-2-13b-chat-hf","CodeLlama-13b-Instruct-hf"]:
        new_prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

{ user_message } [/INST]
"""
        return new_prompt
    
    # tuluseries
    if model_name == "tulu-2-13b":
        new_prompt = f"""<|user|>
{ user_message }
<|assistant|>
"""
        return  new_prompt

    # zephyr-7b-beta
    if model_name == "zephyr-7b-beta":
        new_prompt = f"""<|system|>
you are a helpful assistant.</s>
<|user|>
{ user_message }</s>
<|assistant|>
"""
        return new_prompt
    
    # mpt-7b-chat
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
    
    if model_name == "Meta-Llama-3-8B-Instruct":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},]
        new_prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True)
        return new_prompt
        
    return user_message

    

def get_response(prompt):

    global pipeline,tokenizer

    user_message = prompt
    prompt = format_prompt(prompt)


    if user_message == prompt:
        print("The model you are using does not have a prompt template!")
        sys.exit()
    print("\nThe above prompt has been formatted according to the model specific chat template.")
    
    terminators = tokenizer.eos_token_id
    # llama3
    if model_name == "Meta-Llama-3-8B-Instruct":
        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature = 0.75,
        num_return_sequences=1,
        eos_token_id=terminators,
        max_new_tokens=4096, 
    )

    for seq in sequences:
        # print(f"Result: {seq['generated_text']}")
        context = seq['generated_text'] 
        only_response = context.split(prompt, 1)[-1]
        return only_response


# Replace the part of the prompt designed for chatGPT and change the name of LLM to the specific test model name
def change_LLM_name_in_prompt(old_prompt,LLM_name):
    if LLM_name == "chatgpt" :
        return old_prompt
    
    result_str = re.sub('chatgpt', LLM_name, old_prompt, flags=re.IGNORECASE)
    return result_str


def main():

    # logo
    print("""
██████╗ ███╗   ███╗ ██████╗██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗████╗ ████║██╔════╝██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██████╔╝██╔████╔██║██║     ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██╗██║╚██╔╝██║██║     ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██║  ██║██║ ╚═╝ ██║╚██████╗██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                                                                     
          """)
    
    global model_info_dict,pipeline,tokenizer,model_name
    

    parser = argparse.ArgumentParser(description='parsers')
    parser.add_argument('--run_model_index',type=int,required=True, help='model_index')
    parser.add_argument('--run_rounds', type=int,nargs='+',required=True, help='The number of rounds for this experiment (can be multiple rounds), with numbers separated by spaces')

    # 
    args = parser.parse_args()


    run_model_index = args.run_model_index
    run_rounds = args.run_rounds

    print("run_rounds",run_rounds)
    for run_round in run_rounds:
        print(f"the {run_round} round")

        file_path = f'../data/prompt_round_{run_round}.xlsx' # prompt file
        model_info = model_info_dict[run_model_index]
        model_path = "models/"+ model_info["dir_name"] 
        output_file_dir = 'RMCBench/res/'+model_info["dir_name"]+f'/round{ run_round }/' 
        LLM_name = model_info["LLM_name"] 

        print("******************************Key parameters for this run:******************************")
        print("model_path:",model_path)
        print("output_file_dir:",output_file_dir)
        print("LLM_name:",LLM_name)
        print("****************************************************************************")


        if os.path.exists(output_file_dir):
        
            print(f"Output Directory {output_file_dir} exist")
        else:
            print(f"Output Directory {output_file_dir} not exist, create...")
            os.makedirs(output_file_dir)
            print("creat done!")

        model_name = model_path.split('/')[-1] 

        print("loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        )
        print("Loading completed!")
        print("🤩🤩🤩🤩🤩 "+model_name+",go!!!🤩🤩🤩🤩🤩")
        # ===============================================================================

        
        start_time = time.perf_counter()
        
        prompt_df = pd.read_excel(file_path)

        # ******************debug sample test**********************************************************************************************************************************************
        # prompt_df = prompt_df.sample(n=10)
        
        # prompt_df = prompt_df.reset_index(drop=True)
        # *****************************************************************************************************************************************************************************


        prompt_df['response'] = ''

        # samples num
        num_rows = len(prompt_df)
        print("total row:", num_rows)

        index = 0  
        same_error_times=0 
        error_id_list = [] 

        while index < num_rows:
            # if index >=5:
            #     break

            print(f"now is processing the index：{index}  data=============================")
            

            row = prompt_df.iloc[index]
            print("pid:",row['pid'])


            prompt = row['prompt']


            if row['category']=="text-to-code" and row["level"]==3:
                prompt = change_LLM_name_in_prompt(prompt,LLM_name)
                prompt_df.at[index, 'prompt'] = prompt 
                print("Complete jailbreak template replacement")

            print("\nprompt:")
            print(prompt)

            try:
                response = get_response(prompt)
                same_error_times=0
            except Exception as e:
                same_error_times+=1

                print("error: " + str(e))
                print("back to last index")

                if same_error_times>=3 :
                    print(f'the index {index} data error count has reached its maximum value')

                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    output_file = output_file_dir + f"res_{model_name}_{timestamp}_round{run_round}_err_pid_{row['pid']}.xlsx"
                    prompt_df.to_excel(output_file, index=False)
                    
                    error_id_list.append(index)
                    index += 1 

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
        print("save to:",output_file)

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

        print("done! output save to:", output_file_dir)
    
    print("All rounds of all models have been executed!")

if __name__ == "__main__":
    main()

