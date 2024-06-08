import os
import requests
import time
import random
import pandas as pd
import io
import json
import argparse

# 调用模型获得回答
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
        return response_json['choices'][0]['message']['content'] if response_json['choices'] else "无法获取分析结果"
    except Exception as e:
        print(f"OpenAI API 请求出错: {e}")
        return False
    
# 返回一个dataframe
def read_prompt_from_xlxs_file(file_path):
    prompt_df = pd.read_excel(file_path,dtype=str)
    prompt_df['response'] = '' # 增加一列保存gpt回复
    return prompt_df

# 将df保存为文件
def save_res_file(prompt_df,output_file_dir,file_name,model_name):

    timestamp = time.strftime("%Y%m%d%H%M%S")

    if file_name == "tmp":
        timestamp = ""
    
    output_file = output_file_dir + f"res_{ model_name }_{timestamp}_{file_name}.xlsx"
    prompt_df.to_excel(output_file, index=False)
    print("结果文件已保存至：",output_file)

# 主函数
def main():
    """ 主函数 """
    # logo
    print("""
███╗   ███╗ ██████╗ ██████╗  █████╗ ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
████╗ ████║██╔════╝██╔════╝ ██╔══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██╔████╔██║██║     ██║  ███╗███████║██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██║╚██╔╝██║██║     ██║   ██║██╔══██║██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██║ ╚═╝ ██║╚██████╗╚██████╔╝██║  ██║██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                                                               
          """)
    
    parser = argparse.ArgumentParser(description='通过参数指定测试的模型、以及实验轮数')
    parser.add_argument('--run_model_index',type=int,required=True, help='运行模型的索引')
    parser.add_argument('--run_rounds', type=int,nargs='+',required=True, help='本次实验是第几轮(可以是多轮)，数字以空格分割')

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


    print("一共需要进行的轮数：",run_rounds)

    for run_round in run_rounds:
        print(f"第{run_round}轮实验")

        prompt_file_path = f'D:\桌面\BaiduSyncdisk\RMCBench\data\prompt_v2\prompt_round_{run_round}.xlsx'
        model_info = model_info_dict[run_model_index]
        model_name = model_info["dir_name"]
        api_key = "your_api_key"  # open ai api key
        output_file_dir = 'D:/桌面/BaiduSyncdisk/RMCBench/res/'+model_name+f'/round{ run_round }/' # 输出文件路径,需要包含最后的/

        print("******************************本次运行关键参数:******************************")
        print("model_name:",model_name)
        print("output_file_dir:",output_file_dir)
        # print("LLM_name:",model_name)
        print("****************************************************************************")

        # 检查输出路径
        if os.path.exists(output_file_dir):
            print(f"输出目录 {output_file_dir} 存在。")
        else:
            print(f"输出目录 {output_file_dir} 不存在,进行创建")
            os.makedirs(output_file_dir)
            print("创建完成！")

        print("🤩🤩🤩🤩🤩 "+model_name+",启动！！！🤩🤩🤩🤩🤩")
                
        # 开始时间
        start_time = time.perf_counter()
        
        prompt_df = read_prompt_from_xlxs_file(prompt_file_path)

        # ******************debug 抽样测试***********************************************************************************************
        # prompt_df = prompt_df.sample(n=5)
        # # 重置索引，新的dataframe索引也是从0开始
        # prompt_df = prompt_df.reset_index(drop=True)
        # ******************************************************************************************************************************

        # 计算样本数
        num_rows = len(prompt_df)
        print("total case: ",num_rows)

        # 处理每一条数据，获取回答
        index = 0
        same_error_times=0 # 初始化同一条样本出现错误的最大次数
        request_fail_index_set = set() # 出错的样本的id
        
        while index < num_rows:
            print("\n==========================================")
            print("now is procesiong index :" + str(index))
            row = prompt_df.iloc[index]
            print("pid is:"+ row["pid"])

            user_prompt = row['prompt']
            user_prompt = r''.join(user_prompt)
            print("\nprompt:")
            print(user_prompt)

            response = get_response(user_prompt,api_key,model_name) # 获取回答
        
            print("\nLLM:")
            print(response)
            print("\n====================================================================================\n")

            prompt_df.at[index, 'response'] = response # 存入dataframe
            index+=1
        
        # =====================================================================
        print("++++++++++++++++++++++++++++++++++已遍历完轮++++++++++++++++++++++++++++++++++")
        print("失败数据的index：",request_fail_index_set)
        print("共",len(request_fail_index_set))
        # print("先保存文件备份，之后继续处理未成功的数据直到全部成功")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        save_res_file(prompt_df,output_file_dir,f"round{run_round}_first_try",model_name)

        # 计算耗时
        end_time = time.perf_counter()
        e2e_inference_time = (end_time-start_time)
        print(f"The time consuming is {e2e_inference_time} s")

        # 结束的具体时间
        timestamp = time.strftime("%Y%m%d%H%M%S")
        # 本次运行的所有信息
        run_info = {
            "model_name":model_name,
            "end_time":timestamp,
            "time consuming":f"{e2e_inference_time} s",
            "error_id_list":list(request_fail_index_set)
        }

        # 信息保存到文件
        with open(output_file_dir + f"run_info_{model_name}_{timestamp}_round{run_round}_done.json", 'w') as f:
            json.dump(run_info, f)

        print(f"round{run_round}运行完成，输出文件已保存到目录：", output_file_dir)
    
    print("所有轮次实验已执行完毕！")


if __name__ == "__main__":
    main()
