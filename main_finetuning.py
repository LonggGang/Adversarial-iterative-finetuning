import json
import os
import pickle
import time

import numpy as np
import torch
from datasets import Dataset as hg_Dataset
from datasets import VerificationMode, load_dataset, concatenate_datasets
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from args import parse_arguments
from utils import set_random_seed , progress_bar



def finetune(args):
    device = args.device

    # Build and save zero-shot model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_checkpoint, local_files_only=True)

    if args.model_precision is None:
        model = AutoModelForCausalLM.from_pretrained(args.pretrain_checkpoint, local_files_only=True).to(
            device
        )
    else:
        if args.model_precision == "float16":
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrain_checkpoint, dtype=torch.float16, local_files_only=True
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.pretrain_checkpoint, dtype=torch.bfloat16, local_files_only=True
            ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # dataset
    print("loading dataset")
    set_random_seed(args.seed)

    def encode(examples):
        encoding = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = encoding.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        encoding["labels"] = labels

        return encoding

    def filter_short_tokenized_rows(row):
        min_length = 50
        tokenized_text = tokenizer(row["text"], truncation=True)
        return len(tokenized_text["input_ids"]) >= min_length
    
    if(args.dataset == "key_value"):
        with open("../ai4privacy_data/my_data_key_value_simple.json", "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        if(args.num_data_points < len(trainset)):
            trainset = trainset.shuffle(seed=42).select(range(args.num_data_points)) 
        trainset = trainset.map(
            encode, batched=True, remove_columns=trainset.column_names
            )

        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(
            encode, batched=True, remove_columns=eval_dataset.column_names
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
        )
    elif(args.dataset == "nvidia_structured"):
        with open("../ai4privacy_data/structured_nvidia_clean_2k.json", "r") as file:
            loaded_list = json.load(file)
        trainset = hg_Dataset.from_dict({"text": loaded_list})
        if(args.num_data_points < len(trainset)):
            trainset = trainset.shuffle(seed=42).select(range(args.num_data_points)) 
        trainset = trainset.map(
            encode, batched=True, remove_columns=trainset.column_names
            )
        eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]
        eval_dataset = eval_dataset.filter(filter_short_tokenized_rows).map(
            encode, batched=True, remove_columns=eval_dataset.column_names
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=default_data_collator
        )
    print("done")
    for i in range(args.iterative_rounds):
        print(f"Iterative round {i+1}/{args.iterative_rounds}")
        print("Start training with User data\n")
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True, 
            collate_fn=default_data_collator,
        )
        # training
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=args.epochs_user * len(train_loader) // args.accumulation_steps,
        )

        accumulation_steps = args.accumulation_steps
        accumulated_steps = 0
        accumulated_loss = 0

        for epoch in range(args.epochs_user):
            model.train()
            for batch in train_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if (accumulated_steps + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress_bar((accumulated_steps + 1) // accumulation_steps , (args.epochs_user * len(train_loader) // accumulation_steps) + 1,
                            "train_loss: %.3f" % accumulated_loss)
                    accumulated_loss = 0
                accumulated_steps += 1

            ### eval
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in eval_loader:
                    inputs = {key: val.to(device) for key, val in batch.items()}
                    outputs = model(**inputs)
                    loss = outputs.loss.item()
                    val_loss += loss
            
            val_loss = val_loss / len(eval_loader)
            print(f"Epoch {epoch+1}/{args.epochs_user}, Validation Loss: {val_loss}")
        
        if(args.iterative_rounds == 1):
            break
        model.eval()
        total_generated_text = []
        keys = ["Email", "IP Address", "SSN", "Credit Card", "Phone"]
        for key in keys:
            inputs = tokenizer(key, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=32,
                do_sample=True,
                top_k=40,
                top_p=1,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.eos_token_id,
            )

            output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_texts = [text + tokenizer.eos_token for text in output_texts]
            total_generated_text.extend(output_texts)
        
        new_data = {"text": total_generated_text}
        new_dataset = hg_Dataset.from_dict(new_data)
        new_dataset = new_dataset.map(
            encode, batched=True, remove_columns=new_dataset.column_names
        )
        train_loader = torch.utils.data.DataLoader(
            new_dataset,
            batch_size= args.batch_size,
            shuffle=True, 
            collate_fn=default_data_collator,
        )
        # training with newdata
        print("Start training with Newdata\n")
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=args.epochs_gen * len(train_loader) // args.accumulation_steps,
        )

        accumulation_steps = args.accumulation_steps
        accumulated_steps = 0
        accumulated_loss = 0

        for epoch in range(args.epochs_gen):
            model.train()
            for batch in train_loader:
                inputs = {key: val.to(device) for key, val in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss = loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if (accumulated_steps + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    progress_bar((accumulated_steps + 1) // accumulation_steps , (args.epochs_gen * len(train_loader) // accumulation_steps) + 1,
                            "train_loss: %.3f" % accumulated_loss)
                    accumulated_loss = 0
                accumulated_steps += 1

            ### eval
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in eval_loader:
                    inputs = {key: val.to(device) for key, val in batch.items()}
                    outputs = model(**inputs)
                    loss = outputs.loss.item()
                    val_loss += loss
            
            val_loss = val_loss / len(eval_loader)
            print(f"Epoch {epoch+1}/{args.epochs_gen}, Validation Loss: {val_loss}") 
    
    os.makedirs("./saved_iterative_finetune_models/" + args.name, exist_ok=True)
    checkpoint_path = "./saved_iterative_finetune_models/" + args.name + "/"
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)

if __name__ == "__main__":
    args = parse_arguments()
    finetune(args)
