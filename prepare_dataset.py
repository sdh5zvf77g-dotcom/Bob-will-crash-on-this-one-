#!/usr/bin/env python3
"""Prepare a conservative SFT dataset from Crazy Bob's private memory export."""
import argparse, json, re, random
from pathlib import Path

POS = re.compile(r"\b(that worked|it worked|works now|fixed it|that helped|very helpful|that's correct|that is correct|exactly right|good answer|well done|perfect)\b", re.I)
NEG = re.compile(r"\b(didn't work|does not work|doesn't work|still broken|that's wrong|that is wrong|incorrect|not correct|you are wrong|you got that wrong|failed again)\b", re.I)
SECRET = [re.compile(r"(?i)\b(api[_ -]?key|secret|password|bearer)\s*[:=]\s*\S+"), re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), re.compile(r"\b\d{13,19}\b")]
def clean(s):
    s = str(s or '').strip()
    for pat in SECRET: s = pat.sub('[REDACTED]', s)
    s = re.sub(r'<think>[\s\S]*?</think>', '', s, flags=re.I).strip()
    return s[:6000]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--out',required=True); ap.add_argument('--min-answer-chars',type=int,default=40); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
    raw=json.loads(Path(a.input).read_text(encoding='utf-8'))
    msgs=raw.get('messages',[])
    if not isinstance(msgs,list): raise SystemExit('Input export has no messages array')
    examples=[]; rejected=[]; seen=set()
    for i,m in enumerate(msgs):
        if not isinstance(m,dict) or m.get('role')!='assistant': continue
        answer=clean(m.get('content',''))
        if len(answer)<a.min_answer_chars: continue
        # Only explicitly positive feedback directly after an assistant turn qualifies.
        nxt=msgs[i+1] if i+1<len(msgs) else {}
        if not isinstance(nxt,dict) or nxt.get('role')!='user': continue
        feedback=clean(nxt.get('content',''))
        prev=msgs[i-1] if i>0 else {}
        user=clean(prev.get('content','')) if isinstance(prev,dict) and prev.get('role')=='user' else ''
        if NEG.search(feedback):
            rejected.append({'reason':'negative_feedback','prompt':user[:1000],'answer':answer[:1000]}); continue
        if not POS.search(feedback): continue
        if not user or len(user)<4 or '[REDACTED]' in user or '[REDACTED]' in answer: 
            rejected.append({'reason':'review_for_privacy_or_missing_prompt','prompt':user[:1000],'answer':answer[:1000]}); continue
        key=(user.lower(),answer.lower())
        if key in seen: continue
        seen.add(key); examples.append({'messages':[{'role':'user','content':user},{'role':'assistant','content':answer}]})
    if len(examples)<2: raise SystemExit(f'Only {len(examples)} positive-feedback examples found. Need at least 2 to split; preferably 20+ reviewed examples. Check your export and feedback history.')
    random.Random(a.seed).shuffle(examples)
    nval=max(1,round(len(examples)*0.15)) if len(examples)>=7 else 1
    val=examples[:nval]; train=examples[nval:]
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    for name,data in [('train.jsonl',train),('validation.jsonl',val),('review.jsonl',examples),('excluded.jsonl',rejected)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data),encoding='utf-8')
    (out/'DATASET_REPORT.txt').write_text(f'Positive feedback examples: {len(examples)}\nTrain: {len(train)}\nValidation: {len(val)}\nExcluded/review: {len(rejected)}\n\nReview review.jsonl manually for privacy, correctness, duplicates and quality before training. Automated redaction is imperfect.\n',encoding='utf-8')
    print((out/'DATASET_REPORT.txt').read_text())
if __name__=='__main__': main()
