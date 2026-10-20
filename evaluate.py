#!/usr/bin/env python3
"""Generate repeatable outputs for side-by-side human evaluation."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tasks',required=True);p.add_argument('--out',required=True);p.add_argument('--adapter');p.add_argument('--max-new-tokens',type=int,default=220);a=p.parse_args()
 tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
 device='cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
 model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype='auto',low_cpu_mem_usage=True).to(device)
 if a.adapter: model=PeftModel.from_pretrained(model,a.adapter).to(device)
 model.eval(); rows=[json.loads(x) for x in Path(a.tasks).read_text().splitlines() if x.strip()]; out=[]
 for row in rows:
  messages=row.get('messages') or [{'role':'user','content':row['prompt']}]
  prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  ins=tok(prompt,return_tensors='pt').to(model.device)
  with torch.no_grad(): ids=model.generate(**ins,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
  answer=tok.decode(ids[0][ins['input_ids'].shape[1]:],skip_special_tokens=True).strip()
  out.append({'id':row.get('id',str(len(out)+1)),'prompt':messages[-1]['content'],'answer':answer,'adapter':bool(a.adapter)})
 Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8');print(f'Wrote {len(out)} outputs to {a.out}')
if __name__=='__main__':main()
