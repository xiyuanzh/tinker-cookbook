from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import pandas as pd
import time
import numpy as np
from datetime import datetime
import os
import subprocess
import tempfile
import shutil
import re
import sys
import argparse
from tqdm import tqdm

_code_counter = 0
root_folder = None

def _log_error(code_id: int, error_msg: str, stderr: str):
    """Log execution errors to file"""
    log_dir = os.path.join(root_folder, "execution_logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, f"error_{code_id:04d}.txt"), 'w') as f:
        f.write(f"Error: {error_msg}\n")
        f.write(f"Stderr: {stderr}\n")

def execute_code_safely(code, historical_data, future_count, train_dir, val_dir, timeout: int = 30):
    """Execute generated code safely and return predictions"""
    global _code_counter
    _code_counter += 1
    
    codes_dir = os.path.join(root_folder, "generated_codes")
    os.makedirs(codes_dir, exist_ok=True)
    with open(os.path.join(codes_dir, f"code_{_code_counter:04d}.py"), 'w') as f:
        f.write(code)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            code_file = os.path.join(temp_dir, "forecast_code.py")
            with open(code_file, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                [sys.executable, code_file, "--train_dir", train_dir, "--val_dir", val_dir],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            submission_path = os.path.join(temp_dir, "submission.csv")

            if os.path.exists(submission_path):
                submission_dir = os.path.join(root_folder, "generated_submission")
                os.makedirs(submission_dir, exist_ok=True)
                shutil.copy2(submission_path, os.path.join(submission_dir, f"submission_{_code_counter:04d}.csv"))

                try:
                    pred_df = pd.read_csv(submission_path)
                    predictions = pred_df.iloc[:, -1].values
                    
                    if len(predictions) != future_count:
                        if len(predictions) > future_count:
                            predictions = predictions[:future_count]
                        else:
                            pad_value = predictions[-1] if len(predictions) > 0 else np.mean(historical_data) if historical_data else 0.0
                            predictions = np.concatenate([
                                predictions, 
                                [pad_value] * (future_count - len(predictions))
                            ])
                        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                            predictions[np.isnan(predictions)] = 0.0
                            predictions[np.isinf(predictions)] = 0.0
                        return 0.5, predictions, "Prediction length mismatch, adjusted"

                    elif np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                        predictions[np.isnan(predictions)] = 0.0
                        predictions[np.isinf(predictions)] = 0.0
                        return 0.8, predictions, "Predictions contain NaN or Inf"
                    
                    else:
                        return 1.0, predictions, "Success"
                    
                except Exception as e:
                    error_msg = f"Failed to read predictions: {str(e)}"
                    _log_error(_code_counter, error_msg, result.stderr)
                    return 0.1, None, error_msg
            else:
                error_msg = f"No submission file created. stderr: {result.stderr[:200]}"
                _log_error(_code_counter, error_msg, result.stderr)
                return 0.1, None, error_msg
                
    except subprocess.TimeoutExpired:
        error_msg = "Code execution timed out"
        _log_error(_code_counter, error_msg, "")
        return 0, None, error_msg
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        _log_error(_code_counter, error_msg, "")
        return 0, None, error_msg

def calculate_forecasting_reward(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate reward based on forecasting accuracy"""
    assert len(predictions) == len(ground_truth), "Prediction and ground truth lengths do not match"
    
    mse = np.mean((predictions - ground_truth) ** 2)
    scale_factor = 10000  
    accuracy_reward = max(0, 1 - mse / scale_factor)
    
    return accuracy_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='Path to parquet dataset')
    parser.add_argument('--root_folder', required=True, help='Root folder name for outputs')
    args = parser.parse_args()
    
    global root_folder
    root_folder = args.root_folder
    
    # load data from parquet file
    data_path = os.path.join(args.data_path, "test_0028.parquet")
    df = pd.read_parquet(data_path)
    mse_list = []
    mae_list = []
    total_score_list = []

    # model_path = "Qwen/Qwen3-32B" # zero shot
    # model_path = "/fsx/checkpoints/0024" 
    # model_path = "agentica-org/DeepSWE-Preview"
    model_path = "/home/ubuntu/tinker-cookbook/merged_model"
    
    batch_size = 32  

    # load vLLM model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=8,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True
    )

    # sampling parameters
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=1
    )

    # load tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # prepare all prompts for batch processing
    all_texts = []
    all_data = []
    
    for i in range(len(df)):

        train_dir = df['train_dir'].iloc[i]
        val_dir = df['val_dir'].iloc[i]
        prompt = df['extra_info'].iloc[i]["question"]
        future_df = df["extra_info"].iloc[i]["future_data"]
        future_count = len(future_df)
        historical_data = df["extra_info"].iloc[i]["past_data"]
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        all_texts.append(text)
        all_data.append({
            'train_dir': train_dir,
            'val_dir': val_dir,
            'future_df': future_df,
            'future_count': future_count,
            'historical_data': historical_data,
            'index': i
        })
    
    # process in batches
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    print(f"Processing {len(all_texts)} samples in {total_batches} batches of size {batch_size}")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_texts))
        
        batch_texts = all_texts[start_idx:end_idx]
        batch_data = all_data[start_idx:end_idx]
        
        print(f"Batch {batch_idx + 1}/{total_batches}: generating {len(batch_texts)} samples...")
        start_time = time.time()
        outputs = llm.generate(batch_texts, sampling_params)
        end_time = time.time()
        generation_time = end_time - start_time
        print(f"Batch generation time: {generation_time:.2f}s ({generation_time/len(batch_texts):.2f}s per sample)")
        
        # process batch results
        for output, data in tqdm(zip(outputs, batch_data), desc=f"Processing batch {batch_idx + 1}"):
            generated_text = output.outputs[0].text
            
            try:
                think_end = generated_text.rfind("</think>")
                if think_end != -1:
                    thinking_content = generated_text[:think_end].replace("<think>", "").strip()
                    content = generated_text[think_end + 8:].strip()
                else:
                    thinking_content = ""
                    content = generated_text.strip()
            except:
                thinking_content = ""
                content = generated_text.strip()

            python_blocks = re.findall(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
            if python_blocks:
                content = python_blocks[0]
            
            exe_reward, predictions, error_msg = execute_code_safely(
                content, data['historical_data'], data['future_count'], 
                data['train_dir'], data['val_dir']
            )

            if predictions is None:
                accuracy_reward = 0.0
                predictions = np.zeros(data['future_count'])
            else:
                try:
                    accuracy_reward = calculate_forecasting_reward(predictions, data['future_df'])
                except Exception as e:
                    accuracy_reward = 0.0
                    predictions = np.zeros(data['future_count'])
            
            mse = np.mean((predictions - data['future_df']) ** 2)
            mae = np.mean(np.abs(predictions - data['future_df']))
            total_score = 0.5 * exe_reward + 0.5 * accuracy_reward

            mse_list.append(mse)
            mae_list.append(mae)
            total_score_list.append(total_score)

            with open(f"{root_folder}/score.txt", "a") as f:
                f.write(f"{data['index']+1:04d}: {mse}\n")

    # compute avg
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)
    avg_total_score = np.mean(total_score_list)
    print(f"\nFinal Results:")
    print(f"avg mse: {avg_mse:.4f}")
    print(f"avg mae: {avg_mae:.4f}")
    print(f"avg total score: {avg_total_score:.4f}")
    
    # Log results to file
    with open(f"{root_folder}/final_results.txt", "w") as f:
        f.write(f"Final Results:\n")
        f.write(f"avg mse: {avg_mse:.4f}\n")
        f.write(f"avg mae: {avg_mae:.4f}\n")
        f.write(f"avg total score: {avg_total_score:.4f}\n")

if __name__ == '__main__':
    main()