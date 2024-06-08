import os
import requests
import time
import random
import pandas as pd
import io
import json
import argparse

def get_response(user_prompt, api_key, model_name):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    data = {

        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response_json = response.json()
        return response_json['choices'][0]['message']['content'] if response_json['choices'] else ""
    except Exception as e:
        return False
    
def read_prompt_from_xlxs_file(file_path):
    prompt_df = pd.read_excel(file_path,dtype=str)
    prompt_df['response'] = ''
    return prompt_df

def save_res_file(prompt_df,output_file_dir,file_name,model_name):

    timestamp = time.strftime("%Y%m%d%H%M%S")

    if file_name == "tmp":
        timestamp = ""
    
    output_file = output_file_dir + f"res_{ model_name }_{timestamp}_{file_name}.xlsx"
    prompt_df.to_excel(output_file, index=False)

def main():
   
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--run_model_index',type=int,required=True, help='index')
    parser.add_argument('--run_rounds', type=int,nargs='+',required=True, help='rounds')

    args = parser.parse_args()
    
    run_model_index = args.run_model_index
    run_rounds = args.run_rounds
    run_model_index = 0
    run_rounds = [2,3,4,5]

    model_info_dict = [
    {
        "mid":0,
        "dir_name":"gpt-3.5-turbo",
        "LLM_name":"ChatGPT"
    },
    {
        "mid":1,
        "dir_name":"gpt-4",
        "LLM_name":"ChatGPT"
    }]

    for run_round in run_rounds:

        prompt_file_path = f''
        model_info = model_info_dict[run_model_index]
        model_name = model_info["dir_name"]
        api_key = "your_api_key"  # open ai api key
        output_file_dir = '' # 
                
        start_time = time.perf_counter()
        
        prompt_df = read_prompt_from_xlxs_file(prompt_file_path)

        num_rows = len(prompt_df)
        print("total case: ",num_rows)

        index = 0
        same_error_times=0 
        request_fail_index_set = set()
        
        while index < num_rows:
            print("\n==========================================")
            print("now is procesiong index :" + str(index))
            row = prompt_df.iloc[index]
            print("pid is:"+ row["pid"])

            user_prompt = row['prompt']
            user_prompt = r''.join(user_prompt)
            print("\nprompt:")
            print(user_prompt)

            response = get_response(user_prompt,api_key,model_name) 
        
            print("\nLLM:")
            print(response)
            print("\n====================================================================================\n")

            prompt_df.at[index, 'response'] = response 
            index+=1
        
        save_res_file(prompt_df,output_file_dir,f"round{run_round}_first_try",model_name)

        end_time = time.perf_counter()
        e2e_inference_time = (end_time-start_time)
        print(f"The time consuming is {e2e_inference_time} s")

        timestamp = time.strftime("%Y%m%d%H%M%S")
        run_info = {
            "model_name":model_name,
            "end_time":timestamp,
            "time consuming":f"{e2e_inference_time} s",
            "error_id_list":list(request_fail_index_set)
        }

        with open(output_file_dir + f"run_info_{model_name}_{timestamp}_round{run_round}_done.json", 'w') as f:
            json.dump(run_info, f)


if __name__ == "__main__":
    main()
