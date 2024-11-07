# almost entirely lifted from: https://github.com/jxmorris12/vec2text/blob/e5022f169fa971a16d6d5f4788b024c53997debe/vec2text/run_args.py

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DataArguments:
    """arguments for data"""

    dataset_name: Optional[str] = field(
        default="",
        metadata={
            "choices": ["mnist"],  # no choice really haha
            "help": "Name of the dataset used",
        },
    )

    max_eval_samples: int = field(
        default=100,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    use_less_data: int = field(
        default=-1,
        metadata={
            "help": {"Use a small amount of the training/eval data (for testing)"}
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name.")


@dataclass
class TrainingArguments:
    """args for training"""

    output_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for training saves. If not set, will output to saves/<random hash>."
        },
    )
    steps_per_epoch: int = field(
        default=5000,
        metadata={"required": False, "help": "Size of pseudo-training set."},
    )
    num_train_epochs: float = field(
        default=30.0,
        metadata={"required": False, "help": "Number of epochs for training"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    use_wandb: Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to log to Weights & Biases."}
    )
    per_device_train_batch_size: int = field(
        default=64, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": ("Whether to use bf16 (mixed) precision instead of 32-bit.")},
    )
    torch_compile: bool = True

    # experiment settings -- for now only one experiment
    experiment: str = (
        field(
            default="ssl_experiment_concrete_kl",
            metadata={
                "required": False,
                "help": "Which expt to run (defines model, loss func, dataset, hypothesis)...",
                "choices": [
                    "ssl_experiment_concrete_kl",
                    "ssl_experiment_softmax_probs",  # straight through poole
                ],
            },
        ),
    )
    exp_name: str = field(
        default="",
        metadata={
            "required": False,
            "help": "Name to identify this specific run of an experiment",
        },
    )

    # eval
    # Do evaluation and logging on certain num steps.
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    save_strategy: str = "steps"

    save_total_limit: int = 2  # Maximum number of checkpoints to save.

    warmup_steps: int = field(
        default=400, metadata={"help": "Number of steps of warmup"}
    )
    logging_steps: int = field(
        default=400, metadata={"help": "Number of steps between logging metrics"}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps per save"},
    )
    eval_steps: int = field(
        default=400,
        metadata={
            "help": "Number of steps between eval (will be scaled as if batch size is 32)"
        },
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "When using distributed training, the value of the flag `find_unused_parameters` passed to "
                "`DistributedDataParallel`."
            )
        },
    )

    include_inputs_for_metrics: bool = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

    def __post__init__(self):
        super().__post__init__()
        self._frozen = True
        self.report_to = (
            ["wandb"] if (self.use_wandb and (self.local_rank <= 0)) else []
        )
        self.dataloader_pin_memory = True
        num_workers = torch.cuda.device_count()
        self.dataloader_num_workers = num_workers
        print(f"Set num workers to {num_workers}")
        self.dataloader_drop_last = False

        # Scale logging steps proportional to batch size.
        self.warmup_steps = round(self.warmup_steps * (32 / self.train_batch_size))
        self.logging_steps = round(self.logging_steps * (32 / self.train_batch_size))
        self.eval_steps = round(self.eval_steps * (32 / self.train_batch_size))
        self.save_steps = round(self.save_steps * (32 / self.train_batch_size))

        self.adam_epsilon = 1e-6

        self.load_best_model_at_end = True
        self.greater_is_better = False

        self.do_eval = False
