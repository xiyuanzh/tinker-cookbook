import math
import ast
import pandas as pd
import numpy as np
import subprocess
import tempfile
import os
import sys
import re
import shutil
from functools import partial
from typing import Literal, Sequence, cast
from pathlib import Path

import chz
from datasets import Dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer
import tinker

_global_code_counter = 0
root_folder = "/tmp/tinker-examples/rl_basic_ts_2/"


def _log_error(code_id: int, error_msg: str, stderr: str):
    os.makedirs(root_folder + "execution_logs", exist_ok=True)
    with open(root_folder + f"execution_logs/error_{code_id:04d}.txt", 'w') as f:
        f.write(f"Error: {error_msg}\n")
        f.write(f"Stderr: {stderr}\n")

class MMTSEnv(ProblemEnv):
    def __init__(
        self,
        prompt: str,
        past_data: list[float],
        future_data: list[float],
        train_dir: str,
        val_dir: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        super().__init__(renderer, convo_prefix)
        self.prompt = prompt
        self.past_data = past_data
        self.future_data = future_data
        self.train_dir = train_dir
        self.val_dir = val_dir
        global _global_code_counter
        _global_code_counter += 1
        self.code_id = _global_code_counter

    def get_question(self) -> str:
        return self.prompt

    def check_format(self, sample_str: str) -> bool:
        # Check if output contains Python code block
        return bool(re.search(r'```python\s*\n.*?\n```', sample_str, re.DOTALL))

    def check_answer(self, sample_str: str) -> bool:
        # Required abstract method, but not used since we override step()
        return True
    
    async def step(self, action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        correct_format = float(parse_success) and float(self.check_format(message["content"]))
        
        # Execute code and get detailed rewards
        exe_reward, predictions, _ = self._execute_code(message["content"])
        if predictions is not None:
            accuracy_reward = self._calculate_forecasting_reward(predictions, np.array(self.future_data), np.array(self.past_data))
        else:
            accuracy_reward = 0.0
            predictions = np.zeros(len(self.future_data))
        
        total_reward = 0.5 * exe_reward + 0.5 * accuracy_reward
        
        # Calculate MSE and MAE
        mae = float(np.mean(np.abs(predictions - np.array(self.future_data))))
        mse = float(np.mean((predictions - np.array(self.future_data)) ** 2))
        
        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "execution_reward": exe_reward,
                "accuracy_reward": accuracy_reward,
                "total_score": total_reward,
                "format": correct_format,
                "mae": mae,
                "mse": mse,
            },
        )
    
    def _clean_generated_code(self, code: str) -> str:
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', code, re.DOTALL)
        if python_blocks:
            code = python_blocks[0]
        return code.strip()
    
    def _execute_code(self, code: str, timeout: int = 300):

        os.makedirs(root_folder + "generated_outputs", exist_ok=True)
        with open(root_folder + f"generated_outputs/output_{self.code_id:04d}.txt", 'w') as f:
            f.write(code)

        code = self._clean_generated_code(code)
        
        # Log generated code
        os.makedirs(root_folder + "generated_codes", exist_ok=True)
        with open(root_folder + f"generated_codes/code_{self.code_id:04d}.py", 'w') as f:
            f.write(code)
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                code_file = os.path.join(temp_dir, "forecast_code.py")
                with open(code_file, 'w') as f:
                    f.write(code)
                
                result = subprocess.run(
                    [sys.executable, code_file, "--train_dir", self.train_dir, "--val_dir", self.val_dir],
                    cwd=temp_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                submission_path = os.path.join(temp_dir, "submission.csv")
                if os.path.exists(submission_path):
                    # Log submission
                    os.makedirs(root_folder + "generated_submission", exist_ok=True)
                    shutil.copy2(submission_path, root_folder + f"generated_submission/submission_{self.code_id:04d}.csv")
                    
                    pred_df = pd.read_csv(submission_path)
                    predictions = pred_df.iloc[:, -1].values
                    
                    if len(predictions) != len(self.future_data):
                        if len(predictions) > len(self.future_data):
                            predictions = predictions[:len(self.future_data)]
                        else:
                            pad_value = predictions[-1] if len(predictions) > 0 else np.mean(self.past_data)
                            predictions = np.concatenate([
                                predictions, 
                                [pad_value] * (len(self.future_data) - len(predictions))
                            ])
                        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                            predictions[np.isnan(predictions)] = 0.0
                            predictions[np.isinf(predictions)] = 0.0
                        return 0.5, predictions, "Length mismatch"
                    
                    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                        predictions[np.isnan(predictions)] = 0.0
                        predictions[np.isinf(predictions)] = 0.0
                        return 0.8, predictions, "NaN/Inf values"
                    
                    return 1.0, predictions, "Success"
                else:
                    error_msg = f"No submission file created. stderr: {result.stderr[:200]}"
                    _log_error(self.code_id, error_msg, result.stderr)
                    return 0.1, None, error_msg
        except subprocess.TimeoutExpired:
            error_msg = "Code execution timed out"
            _log_error(self.code_id, error_msg, "")
            return 0, None, error_msg
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            _log_error(self.code_id, error_msg, "")
            return 0, None, error_msg
    
    def _calculate_forecasting_reward(self, predictions: np.ndarray, ground_truth: np.ndarray, historical_data: np.ndarray, use_delta: bool = False, delta_weight: float = 0.25) -> float:
        assert len(predictions) == len(ground_truth), "Prediction and ground truth lengths do not match"
        T = len(ground_truth)

        mae = np.mean(np.abs(predictions - ground_truth))
        mse = np.mean((predictions - ground_truth) ** 2)

        hist = np.asarray(historical_data, dtype=float)
        if hist.size == 0:
            # fallback: use ground-truth scale if no history
            scale = np.mean(np.abs(ground_truth)) + 1e-8
        else:
            scale = np.mean(np.abs(hist - hist.mean())) + 1e-8

        norm_mae = mae / scale
        norm_mse = mse / (scale * scale)

        r_mae = np.exp(-norm_mae)
        r_mse = np.exp(-norm_mse)

        base_reward = 0.5 * r_mae + 0.5 * r_mse

        if not use_delta or T < 2:
            return float(np.clip(base_reward, 0.0, 1.0))

        # first differences for predictions and ground truth;
        # anchor deltas at last history point if available, otherwise first gt value
        start_val = hist[-1] if hist.size > 0 else float(ground_truth[0])
        pred_d = np.diff(np.concatenate([[start_val], predictions.astype(float)]))
        gt_d   = np.diff(np.concatenate([[start_val], ground_truth.astype(float)]))

        # scale for deltas: MAD of historical deltas
        if hist.size > 1:
            hist_d = np.diff(hist)
            delta_scale = np.mean(np.abs(hist_d - hist_d.mean())) + 1e-8
        else:
            # if very short history, fall back to level scale
            delta_scale = scale

        delta_mae = np.mean(np.abs(pred_d - gt_d))
        norm_delta_mae = delta_mae / delta_scale
        r_delta = np.exp(-norm_delta_mae)

        delta_w = float(np.clip(delta_weight, 0.0, 1.0))
        base_w = 1.0 - delta_w
        reward = base_w * base_reward + delta_w * r_delta

        return float(np.clip(reward, 0.0, 1.0))

    def get_reference_answer(self) -> str:
        return f"Future values: {self.future_data}"


class MMTSDataset(RLDataset):
    def __init__(
        self,
        parquet_path: str,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        seed: int = 0,
    ):
        self.df = pd.read_parquet(parquet_path)
        if seed > 0:
            self.df = self.df.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def get_batch(self, index: int) -> Sequence[ProblemGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.df))
        assert batch_start < batch_end, "Incorrect batch size"
        
        builders = []
        for i in range(batch_start, batch_end):
            row = self.df.iloc[i]
            builder = self._make_env_group_builder(row, self.group_size)
            if builder is not None:
                builders.append(builder)
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.df) / self.batch_size)

    def _make_env_group_builder(self, row: pd.Series, group_size: int) -> ProblemGroupBuilder | None:
        try:
            extra_info = row['extra_info']
            prompt = extra_info['question']
            past_data = extra_info['past_data']
            future_data = extra_info['future_data']
            train_dir = row['train_dir']
            val_dir = row['val_dir']
            
            # Convert string representations to lists if needed
            if isinstance(past_data, str):
                past_data = ast.literal_eval(past_data)
            if isinstance(future_data, str):
                future_data = ast.literal_eval(future_data)
                
            return ProblemGroupBuilder(
                env_thunk=partial(
                    MMTSEnv,
                    prompt,
                    past_data,
                    future_data,
                    train_dir,
                    val_dir,
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                ),
                num_envs=group_size,
                dataset_name="mmts",
            )
        except Exception as e:
            print(f"Failed to create env group builder: {e}")
            return None


@chz.chz
class MMTSBuilder(RLDatasetBuilder):
    train_parquet_path: str
    eval_parquet_path: str | None = None
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0

    async def __call__(self) -> tuple[MMTSDataset, MMTSDataset | None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        
        train_dataset = MMTSDataset(
            parquet_path=self.train_parquet_path,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            seed=self.seed,
        )
        
        eval_dataset = None
        if self.eval_parquet_path:
            eval_dataset = MMTSDataset(
                parquet_path=self.eval_parquet_path,
                batch_size=self.batch_size,
                group_size=1,  # Use group_size=1 for evaluation
                renderer=renderer,
                seed=0,  # No shuffling for eval
            )
        
        return train_dataset, eval_dataset