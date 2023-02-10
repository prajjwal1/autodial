import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import datasets
#  import numpy as np
from datasets import load_dataset, load_metric

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class ModelArguments:
    model_name_or_path: str
    tokenizer_name: str
    config_name: str
    cache_dir: str
    model_revision: str = field(default='main')
    use_auth_token: bool = field(default=False)
    use_fast_tokenizer: bool = field(default=False)
    ignore_mismatched_sizes: bool = field(default=False)

@dataclass
class DataArguments:
    overwrite_cache: bool = field(default=False)
    max_seq_length: int = field(default=128)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: int = field(default=None)
    max_eval_samples: int = field(default=None)

def hf_parsing():
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args

def load_datasets():
    train_dataset = datasets.load_from_disk("train_data.hf")
    validation_dataset = datasets.load_from_disk("validation_data.hf")
    test_dataset = datasets.load_from_disk("test_data.hf")
    return train_dataset, validation_dataset, test_dataset

logger = logging.getLogger(__name__)

class TransformerPooler(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.dense = nn.Linear(model_args.embedding_size, model_args.embedding_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

from collections.abc import Mapping
from typing import Any, Dict
from transformers.data.data_collator import InputDataClass

def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels_1"] = torch.tensor([f["label_1"] for f in features], dtype=dtype)
        batch["labels_2"] = torch.tensor([f["label_2"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


@dataclass
class ClassifierOutput(SequenceClassifierOutput):
    logits_1: torch.Tensor = None
    logits_2: torch.Tensor = None

class DialogActModel(nn.Module):
    def __init__(self, model_args, config):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_args.model_name_or_path,
                                                   config=config)
        model_args.embedding_size = 1024
        model_args.ffn_size = 4096

        self.linear_1 = nn.Linear(model_args.embedding_size, model_args.ffn_size)
        self.linear_2 = nn.Linear(model_args.ffn_size, model_args.num_labels_1)
        self.linear_3 = nn.Linear(model_args.ffn_size, model_args.num_labels_2)
        self.non_linear = F.relu

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels_1: Tuple[torch.Tensor] = None,
                labels_2: Tuple[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None
                ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        x = outputs[1]
        x = self.non_linear(self.linear_1(x))
        logits_1 = self.linear_2(x)
        loss_1 = loss_fct(logits_1, labels_1.float())
        logits_2 = self.linear_3(x)
        loss_2 = loss_fct(logits_2, labels_2.float())
        loss = loss_1+loss_2

        return ClassifierOutput(
            loss=loss,
            logits_1=logits_1,
            logits_2=logits_2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    train_dataset, validation_dataset, test_dataset = load_datasets()
    model_args, data_args, training_args = hf_parsing()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    finetuning_task = 'multiwoz_v22_dst'

    domain_act_list = [
            'None',
            'Taxi-Request',
            'Police-Inform',
            'Hotel-Inform',
            'Hotel-Request',
            'Police-Request',
            'Hospital-Request',
            'Hospital-Inform',
            'general-greet',
            'Restaurant-Request',
            'Attraction-Inform',
            'Restaurant-Inform',
            'Taxi-Inform',
            'Attraction-Request',
            'general-bye',
            'Train-Inform',
            'general-thank',
            'Train-Request',
        ]
    entity_list = [
            'none',
            'Attraction-Inform_none',
            'Attraction-Inform_type',
            'Attraction-Inform_area',
            'Attraction-Inform_name',
            'Attraction-Inform_entrancefee',
            'Attraction-Request_phone',
            'Attraction-Request_postcode',
            'Attraction-Request_entrancefee',
            'Attraction-Request_name',
            'Attraction-Request_address',
            'Attraction-Request_type',
            'Attraction-Request_area',
            'Attraction-Request_parking',
            'general-bye_none',
            'general-thank_none',
            'general-greet_none',
            'Restaurant-Inform_booktime',
            'Restaurant-Inform_bookday',
            'Restaurant-Request_ref',
            'Restaurant-Request_address',
            'Restaurant-Request_phone',
            'Restaurant-Request_pricerange',
            'Restaurant-Request_postcode',
            'Restaurant-Request_name',
            'Restaurant-Request_area',
            'Restaurant-Inform_none',
            'Restaurant-Inform_food',
            'Restaurant-Inform_pricerange',
            'Restaurant-Inform_bookpeople',
            'Restaurant-Inform_area',
            'Restaurant-Inform_name',
            'Restaurant-Request_food',
            'Hotel-Inform_none',
            'Hotel-Inform_choice',
            'Hotel-Inform_area',
            'Hotel-Inform_bookpeople',
            'Hotel-Inform_internet',
            'Hotel-Inform_bookday',
            'Hotel-Inform-bookpeople',
            'Hotel-Inform_bookstay',
            'Hotel-Inform_parking',
            'Hotel-Inform_pricerange',
            'Hotel-Inform_name',
            'Hotel-Inform_stars',
            'Hotel-Inform_type',
            'Hotel-Request_pricerange',
            'Hotel-Request_parking',
            'Hotel-Request_address',
            'Hotel-Request_name',
            'Hotel-Request_type',
            'Hospital-Inform_none',
            'Hospital-Inform_department',
            'Hospital-Request_phone',
            'Hospital-Request_name',
            'Hospital-Request_postcode',
            'Hospital-Request_address',
            'Hotel-Request_stars',
            'Hotel-Request_ref',
            'Hotel-Request_area',
            'Hotel-Request_internet',
            'Hotel-Request_phone',
            'Hotel-Request_postcode',
            'Train-Inform_none',
            'Train-Inform_day',
            'Train-Inform_departure',
            'Train-Inform_arriveby',
            'Train-Inform_leaveat',
            'Train-Inform_destination',
            'Train-Inform_bookpeople',
            'Train-Inform_price',
            'Train-Request_ref',
            'Train-Request_name',
            'Train-Request_price',
            'Train-Request_trainid',
            'Train-Request_duration',
            'Train-Request_leaveat',
            'Train-Request_arriveby',
            'Taxi-Inform_departure',
            'Taxi-Inform_none',
            'Taxi-Inform_destination',
            'Taxi-Inform_leaveat',
            'Taxi-Inform_arriveby',
            'Taxi-Inform_bookpeople',
            'Taxi-Request_phone',
            'Taxi-Request_name',
            'Taxi-Request_type',
            'Police-Inform_none',
            'Police-Request_name',
            'Police-Request_address',
            'Police-Request_phone',
            'Police-Request_postcode',
            'Police-Request_department'
        ]

    domain_act_dict_label2idx = {
        v: k for k, v in enumerate(domain_act_list)
    }
    domain_act_dict_idx2label = {
        k: v for k, v in enumerate(domain_act_list)
    }
    entity_dict_idx2label = {
        k: v for k, v in enumerate(entity_list)
    }
    entity_dict_label2idx = {
        v: k for k, v in enumerate(entity_list)
    }

    model_args.num_labels_1 = len(domain_act_dict_label2idx)
    model_args.num_labels_2 = len(entity_dict_label2idx)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        #  num_labels=len(classes),
        finetuning_task=finetuning_task,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = DialogActModel(model_args, config)

#      model = AutoModelForSequenceClassification.from_pretrained(
        #  model_args.model_name_or_path,
        #  from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #  config=config,
        #  cache_dir=model_args.cache_dir,
        #  revision=model_args.model_revision,
        #  use_auth_token=True if model_args.use_auth_token else None,
        #  ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
#      )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False


    def preprocess_function(examples):
        text = examples["text"]
        dialog_act_entry = examples["dialog_act"]
        dialog_act_entry = {k:v for k, v in dialog_act_entry.items() if v is not None}

        domain_act_list, entity_list = [], []

        for domain_act_, values in dialog_act_entry.items():
            domain_act_list.append(domain_act_)
            for entity_, _ in values:
                entity_list.append(entity_)

        if not entity_list:
            entity_list.append('none')

        domain_act_indices = [domain_act_dict_label2idx[x] for x in domain_act_list]
        entity_indices = [entity_dict_label2idx[x] for x in entity_list]

        domain_act_multi_hot_label = 0
        for x in domain_act_indices:
            one_hot = F.one_hot(torch.tensor(x), len(domain_act_dict_label2idx))
            domain_act_multi_hot_label += one_hot

        entity_multi_hot_label = 0
        for x in entity_indices:
            one_hot = F.one_hot(torch.tensor(x), len(entity_dict_label2idx))
            entity_multi_hot_label += one_hot

        #  examples.pop('label')  # this is the belief state label (generative)
        examples.pop('dialog_act')
        output = tokenizer(
            text, padding=padding, max_length=max_seq_length, truncation=True
        )
        if torch.cuda.is_available():
            domain_act_multi_hot_label = domain_act_multi_hot_label.cuda()
            entity_multi_hot_label = entity_multi_hot_label.cuda()

        output['labels_1'] = domain_act_multi_hot_label
        output['labels_2'] = entity_multi_hot_label
        return output

    #  train_dataset, validation_dataset, test_dataset = dataset["train"], dataset["validation"], dataset["test"]

    cols_to_remove = ['text', 'dialogue_id', 'episode_done', 'action_intent', 'turn_num', 'id', 'labels']
    eval_cols_to_remove = ['text', 'dialogue_id', 'episode_done', 'action_intent', 'turn_num', 'id', 'eval_labels']

    train_dataset.remove_columns(cols_to_remove)
    validation_dataset.remove_columns(eval_cols_to_remove)
    test_dataset.remove_columns(eval_cols_to_remove)

    train_dataset = train_dataset.map(preprocess_function, batched=False, load_from_cache_file=not data_args.overwrite_cache)
    validation_dataset = validation_dataset.map(preprocess_function, batched=False, load_from_cache_file=not data_args.overwrite_cache)
    test_dataset = test_dataset.map(preprocess_function, batched=False, load_from_cache_file=not data_args.overwrite_cache)

    metric = load_metric("accuracy")

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=test_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_datasets = [validation_dataset, test_dataset]

        for eval_dataset in eval_datasets:
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)



if __name__ == '__main__':
    main()
