#!/usr/bin/env python3
"""Deterministically validate atlas and scope proposals against extracted chunks."""
import hashlib,json,re,sys
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROPOSALS=HERE/'proposals.jsonl'; ALIASES=HERE/'ENTITY_ALIASES.json'
CHUNKS=Path('/tmp/gates-graphrag-pilot/chunks-enriched.jsonl'); REPORT=HERE/'VALIDATION.json'
ALLOWED={'APPLIES_TO','USES_GROUP','USES_ALGEBRA','DESCRIBES_REPRESENTATION','DESCRIBES_MULTIPLET','HAS_INPUT','HAS_OUTPUT','CATALOGS','CLASSIFIES'}
REQUIRED={'proposal_id','source','relationship','target','evidence','basis','review_status','confidence'}
ENTITY_REQUIRED={'type','key','name'}; EVIDENCE_REQUIRED={'paper_id','chunk_id','page_number','section','excerpt'}
PILOT={'1911.00807','2002.08502','2006.03609','2007.07390','2012.13308','2012.14015','2304.09830','2311.06842','2407.09334'}
COUNT_TARGET=re.compile(r'^(?:scope|property):(dimension|spacetime-dimension|supercharge-count|color-count|node-count)(?:$|-)')
def norm(v): return re.sub(r'\s+',' ',v).strip()
def load_jsonl(path):
    with path.open(encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def main():
    errors=[]
    chunks=load_jsonl(CHUNKS) if CHUNKS.exists() else []
    if not chunks: errors.append(f'missing or empty chunk corpus: {CHUNKS}')
    by_id={x['chunk_id']:x for x in chunks}; ps=load_jsonl(PROPOSALS)
    aliases=json.loads(ALIASES.read_text(encoding='utf-8')); ids=[]
    for lineno,p in enumerate(ps,1):
        pid=p.get('proposal_id'); ids.append(pid); label=pid or f'line {lineno}'
        if REQUIRED-set(p): errors.append(f'{label}: missing fields {sorted(REQUIRED-set(p))}')
        if p.get('relationship') not in ALLOWED: errors.append(f'{label}: invalid relationship {p.get("relationship")}')
        if p.get('basis')!='explicit_text': errors.append(f'{label}: basis is not explicit_text')
        if p.get('review_status')!='pending': errors.append(f'{label}: review status is not pending')
        conf=p.get('confidence')
        if not isinstance(conf,(int,float)) or not 0<=conf<=1: errors.append(f'{label}: invalid confidence')
        for side in ('source','target'):
            e=p.get(side,{})
            if ENTITY_REQUIRED-set(e): errors.append(f'{label}: malformed {side}')
            key=e.get('key',''); typ=e.get('type','')
            if typ=='paper':
                if not key.startswith('arxiv:'): errors.append(f'{label}: paper key lacks arxiv prefix')
            elif not key.startswith(typ+':'): errors.append(f'{label}: {side} key prefix mismatch')
        if COUNT_TARGET.match(p.get('target',{}).get('key','')):
            errors.append(f'{label}: dimension/count encoded as relationship target')
        ev=p.get('evidence',{})
        if EVIDENCE_REQUIRED-set(ev): errors.append(f'{label}: malformed evidence')
        ch=by_id.get(ev.get('chunk_id'))
        if not ch: errors.append(f'{label}: missing chunk {ev.get("chunk_id")}'); continue
        if ev.get('paper_id')!=ch.get('paper_id'): errors.append(f'{label}: paper mismatch')
        if ev.get('page_number')!=ch.get('page_number'): errors.append(f'{label}: physical page mismatch')
        if ev.get('section')!=ch.get('section_heading'): errors.append(f'{label}: section mismatch')
        if norm(ev.get('excerpt','')) not in norm(ch.get('text','')): errors.append(f'{label}: excerpt absent after whitespace normalization')
    dup=sorted(k for k,v in Counter(ids).items() if v>1)
    if dup: errors.append(f'duplicate proposal IDs: {dup}')
    covered={p['evidence']['paper_id'] for p in ps}
    if covered!=PILOT: errors.append(f'paper coverage mismatch: missing={sorted(PILOT-covered)}, extra={sorted(covered-PILOT)}')
    alias_keys=[]
    for a in aliases.get('entities',[]):
        alias_keys.append(a.get('key'))
        if not a.get('key') or not a.get('canonical_name') or not a.get('aliases'): errors.append(f'malformed alias: {a}')
    alias_dups=sorted(k for k,v in Counter(alias_keys).items() if v>1)
    if alias_dups: errors.append(f'duplicate alias keys: {alias_dups}')
    report={'schema_version':'gates-scope-atlas-validation-v1','status':'pass' if not errors else 'fail','proposal_count':len(ps),'paper_count':len(covered),'relationship_counts':dict(sorted(Counter(p['relationship'] for p in ps).items())),'paper_counts':dict(sorted(Counter(p['evidence']['paper_id'] for p in ps).items())),'alias_entity_count':len(alias_keys),'proposals_sha256':hashlib.sha256(PROPOSALS.read_bytes()).hexdigest(),'entity_aliases_sha256':hashlib.sha256(ALIASES.read_bytes()).hexdigest(),'errors':errors}
    REPORT.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2,sort_keys=True)); return 1 if errors else 0
if __name__=='__main__': sys.exit(main())
