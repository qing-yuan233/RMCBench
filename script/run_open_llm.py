from transformers import AutoTokenizer
import transformers
import torch
import pandas as pd
import time
# from check_refuse import refuse_generate_malicious_code
from datetime import datetime
import json
import re
import os
import argparse

# 全局变量

# 要测试的8个开源模型的目录和名字
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

model_name = "" # 模型名字(带参数)
pipeline = "" # 模型pipline
tokenizer = "" # 分词器

# 根据不同LLM的prompt chat模板进行格式化
def format_prompt(user_message):
    global model_name
    # llama系列
    if model_name in ["llama-2-7b-chat-hf","llama-2-13b-chat-hf","CodeLlama-13b-Instruct-hf"]:
        new_prompt = f"""<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

{ user_message } [/INST]
"""
        return new_prompt
    
    # tulu系列
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
        
    return user_message

    

# 推理，获取LLM回答
def get_response(prompt):

    global pipeline,tokenizer

    # 根据不同的模型进行格式化
    user_message = prompt
    prompt = format_prompt(prompt)

    # 检查是否格式化，没有进行格式化，则没有正确匹配到模型
    if user_message == prompt:
        print("你所使用的模型没有prompt模板！")
        sys.exit()
    print("\n已将上述prompt按照模型专属chat模板进行格式化")
    
    terminators = tokenizer.eos_token_id
    # llama3,多了一个配置
    
    sequences = pipeline(
        prompt, # 具体的提示词
        do_sample=True,
        top_k=10,
        top_p=0.9,
        temperature = 0.75,
        num_return_sequences=1,
        max_new_tokens=4096, # 生成的最大token数
        # max_length = 4096 # 模型最大token数
    )

    for seq in sequences:
        # print(f"Result: {seq['generated_text']}")
        context = seq['generated_text'] # 完整的输入+回答
        # 只获取模型回答,不包含提问
        only_response = context.split(prompt, 1)[-1]
        return only_response


# 替换prompt中针对chatGPT设计的部分,讲LLM的名字改为具体测试模型的名字
def change_LLM_name_in_prompt(old_prompt,LLM_name):
    if LLM_name == "chatgpt" :
        return old_prompt
    
    result_str = re.sub('chatgpt', LLM_name, old_prompt, flags=re.IGNORECASE)
    return result_str

# 主函数
def main():

    # logo
    print("""
███╗   ███╗ ██████╗ ██████╗  █████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
████╗ ████║██╔════╝██╔════╝ ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██╔████╔██║██║     ██║  ███╗███████║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██║╚██╔╝██║██║     ██║   ██║██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██║ ╚═╝ ██║╚██████╗╚██████╔╝██║  ██║██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                                                               
          """)
    
    global model_info_dict,pipeline,tokenizer,model_name
    
    # 通过命令接受参数
    parser = argparse.ArgumentParser(description='通过参数指定测试的模型、以及实验轮数')
    parser.add_argument('--run_model_index',type=int,required=True, help='运行模型的索引')
    parser.add_argument('--run_rounds', type=int,nargs='+',required=True, help='本次实验是第几轮(可以是多轮)，数字以空格分割')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据输入参数自动生成的关键参数
    run_model_index = args.run_model_index
    run_rounds = args.run_rounds

    print("一共需要进行的轮数：",run_rounds)
    for run_round in run_rounds:
        print(f"第{run_round}轮实验")

        file_path = f'' # 包含prompt的xlxs文件
        model_info = model_info_dict[run_model_index]
        model_path = "" # 模型目录
        output_file_dir = '' # 输出文件路径,需要包含最后的/
        LLM_name = model_info["LLM_name"] # 不带任何参数的LLM模型名，用于替换越狱提示词中的指示对象

        print("******************************本次运行关键参数:******************************")
        print("model_path:",model_path)
        print("output_file_dir:",output_file_dir)
        print("LLM_name:",LLM_name)
        print("****************************************************************************")

        # 检查输出路径
        if os.path.exists(output_file_dir):
            print(f"输出目录 {output_file_dir} 存在。")
        else:
            print(f"输出目录 {output_file_dir} 不存在,进行创建")
            os.makedirs(output_file_dir)
            print("创建完成！")

        model_name = model_path.split('/')[-1] # 模型名字带参数

        # 加载模型和分词器===============================================================
        print("加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        )
        print("加载完成！")
        print("🤩🤩🤩🤩🤩 "+model_name+",启动！！！🤩🤩🤩🤩🤩")
        # ===============================================================================

        # 开始时间
        start_time = time.perf_counter()
        
        # 从文件中获取prompt
        prompt_df = pd.read_excel(file_path)

        # ******************debug 抽样测试**********************************************************************************************************************************************
        # prompt_df = prompt_df.sample(n=10)
        # # 重置索引，新的dataframe索引也是从0开始
        # prompt_df = prompt_df.reset_index(drop=True)
        # *****************************************************************************************************************************************************************************

        # 初始化创建列存储结果的列,以及是否拒绝回答
        prompt_df['response'] = ''

        # 计算样本数
        num_rows = len(prompt_df)
        print("total row:", num_rows)

        index = 0  # 初始化索引
        same_error_times=0 # 初始化同一条样本出现错误的最大次数
        error_id_list = [] # 出错的样本的id

        while index < num_rows:
            # if index >=5:
            #     break

            print(f"now is processing the index：{index}  data=============================")
            
            # 根据索引取到行
            row = prompt_df.iloc[index]
            print("pid:",row['pid'])

            # 获取具体的prompt
            prompt = row['prompt']

            # 如果为越狱，在这里动态替换越狱提示词中指示的模型
            if row['category']=="text-to-code" and row["level"]==3:
                prompt = change_LLM_name_in_prompt(prompt,LLM_name)
                prompt_df.at[index, 'prompt'] = prompt # 替换到结果中
                print("完成越狱词替换")

            print("\nprompt:")
            print(prompt)

            try:
                response = get_response(prompt)
                # 当本条样本提问完成时，错误数清零
                same_error_times=0
            except Exception as e:
                # 增加累计错误数
                print("出错了！")

            # 打印输出数据
            print("\nLLM:")
            print(response)
            print("\n====================================================================================\n")
            # 检查是否拒绝生成

            # 无效数据，通常模型会不停的输出很多空白行，会影响df保存为文件，导致写入数据为空，所以不保留其回答
            response_line = response.count('\n') + 1
            if response_line >=500: # 数据集中最长的代码文件也才347行，所以输出行数基本不可能比这个多
                response  = "Long text exception output.Prevent memory overflow and replace it."
                print(response)

            # 数据更新到df中
            prompt_df.at[index, 'response'] = str(response)

            # 处理成功，移动到下一行
            index += 1

        # 所有prompt推理完成=======================================================
            
        # 最终保存的结果，后缀为done！
        timestamp = time.strftime("%Y%m%d%H%M%S")
        
        output_file = output_file_dir + f"res_{model_name}_{timestamp}_round{run_round}_done.xlsx"
        print("文件保存路径",output_file)

        print("保存为json和xlxs")
        json_name = output_file_dir + f"res_{model_name}_{timestamp}_round{run_round}_done.json"
        prompt_df.to_json(json_name,orient='records')
        prompt_df.to_excel(output_file, index=False)

        # 计算耗时
        end_time = time.perf_counter()
        e2e_inference_time = (end_time-start_time)
        print(f"The time consuming is {e2e_inference_time} s")

        # 本次运行的所有信息
        run_info = {
            "model_name":model_name,
            "run_time":timestamp,
            "time consuming":f"{e2e_inference_time} s",
            "error_id_list":error_id_list
        }
        # 信息保存到文件
        with open(output_file_dir + f"run_info_{model_name}_{timestamp}_round{run_round}_done.json", 'w') as f:
            json.dump(run_info, f)

        print("运行完成，输出文件已保存到目录：", output_file_dir)
    
    print("所有轮次实验已执行完毕！")

if __name__ == "__main__":
    main()

