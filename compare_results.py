#!/usr/bin/env python3
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True);a=p.parse_args()
def load(f):return {x.get('id'):x for x in (json.loads(l) for l in Path(f).read_text().splitlines() if l.strip())}
b,c=load(a.baseline),load(a.candidate)
for k in sorted(set(b)|set(c),key=str):
 print('\n'+'='*90+'\nTASK '+str(k))
 print('\nBASELINE:\n'+b.get(k,{}).get('answer','[missing]'))
 print('\nCANDIDATE:\n'+c.get(k,{}).get('answer','[missing]'))
 print('\nHuman score (0-4): correctness __ / evidence __ / instruction-following __ / safety __ / clarity __ / regressions __')
