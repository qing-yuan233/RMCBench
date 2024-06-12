import os
import requests
import time
import random
import pandas as pd
import io
import json
import argparse

import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_response(user_prompt, api_key, model_name):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        # "model":"gpt-4",
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        response_json = response.json()
        return response_json['choices'][0]['message']['content'] if response_json['choices'] else "error"
    except Exception as e:
        print(f"OpenAI API error: {e}")
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
    print("result save to:",output_file)


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
    
    # parser = argparse.ArgumentParser(description='parsers')
    # parser.add_argument('--run_model_index',type=int,required=True, help='model_index')
    # parser.add_argument('--run_rounds', type=int,nargs='+',required=True, help='The number of rounds for this experiment (can be multiple rounds), with numbers separated by spaces')
    
    # 
    # args = parser.parse_args()

    
    # run_model_index = args.run_model_index
    # run_rounds = args.run_rounds
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


    print("run_rounds",run_rounds)

    for run_round in run_rounds:
        print(f"the {run_round} round")

        prompt_file_path = f'RMCBench\data\prompt_v2\prompt_round_{run_round}.xlsx'
        model_info = model_info_dict[run_model_index]
        model_name = model_info["dir_name"]
        api_key = "your_api_key"  # open ai api key
        output_file_dir = 'RMCBench/res/'+model_name+f'/round{ run_round }/'

        print("******************************Key parameters for this run:******************************")
        print("model_name:",model_name)
        print("output_file_dir:",output_file_dir)
        # print("LLM_name:",model_name)
        print("****************************************************************************")

        if os.path.exists(output_file_dir):
            print(f"Output Directory {output_file_dir} exists")
        else:
            print(f"Output Directory {output_file_dir} not exists,create...")
            os.makedirs(output_file_dir)
            print("creat done!")

        print("🤩🤩🤩🤩🤩 "+model_name+",go!!!🤩🤩🤩🤩🤩")
                
        
        start_time = time.perf_counter()
        
        prompt_df = read_prompt_from_xlxs_file(prompt_file_path)

        # ******************debug sample test***********************************************************************************************
        # prompt_df = prompt_df.sample(n=5)
        
        # prompt_df = prompt_df.reset_index(drop=True)
        # ******************************************************************************************************************************

        # samples num
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

            if response !=False: 
                same_error_times=0
            else:
                same_error_times+=1
                print("error: " + "response == False")
                print("back to last index")

                
                if same_error_times>=10 :
                    print(f'the index {index} data error count has reached its maximum value')

                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    output_file = output_file_dir + f"res_{model_name}_{timestamp}_round{run_round}_err_pid_{row['pid']}.xlsx"
                    prompt_df.to_excel(output_file, index=False)
                    
                    request_fail_index_set.add(index)
                    index += 1 
                    
                
                continue
        
            print("\nLLM:")
            print(response)
            print("\n====================================================================================\n")

            prompt_df.at[index, 'response'] = response 
            index+=1
        
        # =====================================================================
        print("++++++++++++++++++++++++++++++++++over round++++++++++++++++++++++++++++++++++")
        print("request_fail_index_set",request_fail_index_set)
        print("共",len(request_fail_index_set))
        
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

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

        print(f"round{run_round}done! output save to:", output_file_dir)
    
    print("All rounds of all models have been executed!")


if __name__ == "__main__":
    main()
