import asyncio

import chz
import sys
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.mmts_builder import MMTSBuilder
from tinker_cookbook.rl import train


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "Qwen/Qwen3-32B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    builder = MMTSBuilder(
        train_parquet_path="/fsx/chronos-o1/dataset/parquet/train_0028.parquet",
        eval_parquet_path="/fsx/chronos-o1/dataset/parquet/eval_0028.parquet",
        batch_size=32,
        group_size=8,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": "/tmp/tinker-examples/rl_basic_ts_2",
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 8092,
            "save_every": 1,
            "eval_every": 0,
            "wandb_project": "tinker",
            "wandb_name": "mmts"
        }
    )


def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
