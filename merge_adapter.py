#!/usr/bin/env python3
import argparse
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import PeftModel
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--adapter',required=True);p.add_argument('--out',required=True);a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.base,use_fast=True);base=AutoModelForCausalLM.from_pretrained(a.base,torch_dtype='auto',device_map='cpu');model=PeftModel.from_pretrained(base,a.adapter);merged=model.merge_and_unload();merged.save_pretrained(a.out,safe_serialization=True, max_shard_size='2GB');tok.save_pretrained(a.out);print('Merged model saved to',a.out)
