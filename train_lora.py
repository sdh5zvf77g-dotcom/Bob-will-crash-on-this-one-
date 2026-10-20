#!/usr/bin/env python3
"""Conservative LoRA SFT trainer for a compatible Hugging Face causal LM."""
import argparse, json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='meta-llama/Llama-3.2-1B-Instruct'); p.add_argument('--data',required=True); p.add_argument('--validation',required=True); p.add_argument('--out',required=True); p.add_argument('--epochs',type=float,default=2); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--max-length',type=int,default=1024); p.add_argument('--batch-size',type=int,default=1); p.add_argument('--grad-accum',type=int,default=8); a=p.parse_args()
    cuda=torch.cuda.is_available()
    mps=bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    if not cuda and not mps: raise SystemExit('No supported accelerator detected. Use an NVIDIA CUDA GPU or Apple Silicon Mac with PyTorch MPS; CPU training is intentionally not enabled by default.')
    tok=AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    if cuda:
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(a.model, quantization_config=quant, device_map='auto')
        model=prepare_model_for_kbit_training(model)
    else:
        # Apple Silicon: regular LoRA in fp16; bitsandbytes 4-bit QLoRA is CUDA-only.
        model=AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model.to('mps')
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    model=get_peft_model(model,lora); model.print_trainable_parameters()
    ds=load_dataset('json',data_files={'train':a.data,'validation':a.validation})
    def format_row(row):
        # Native chat template keeps the target model's expected message format.
        return {'text':tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False)}
    ds=ds.map(format_row)
    args=TrainingArguments(output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=1, gradient_accumulation_steps=a.grad_accum, gradient_checkpointing=bool(cuda), logging_steps=5, eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, fp16=bool(cuda and not torch.cuda.is_bf16_supported()), bf16=bool(cuda and torch.cuda.is_bf16_supported()), report_to='none', optim='paged_adamw_8bit' if cuda else 'adamw_torch', warmup_ratio=0.05, weight_decay=0.01, seed=42)
    trainer=SFTTrainer(model=model,args=args,train_dataset=ds['train'],eval_dataset=ds['validation'],processing_class=tok,dataset_text_field='text',max_seq_length=a.max_length,packing=False)
    trainer.train(); trainer.save_model(a.out); tok.save_pretrained(a.out)
    metrics=trainer.evaluate(); Path(a.out,'training_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Saved LoRA adapter and metrics to',a.out)
if __name__=='__main__': main()
