#!/usr/bin/env python3
"""Freeze the 400-entry D21 invariant grammar into a compact GPU manifest."""
from __future__ import annotations
import hashlib, json, struct
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXTERNALS=list(range(6))
OUT=set(range(2,6))
OUTER_DEGREES=[0,3,4]
DESCRIPTOR=struct.Struct('<BBBBBBHIhH')
assert DESCRIPTOR.size==16

def valid_pair(a,b): return not (a in OUT and b in OUT)
def matchings(xs):
    if not xs: return [()]
    a=xs[0]; out=[]
    for j in range(1,len(xs)):
        b=xs[j]
        if not valid_pair(a,b): continue
        rem=xs[1:j]+xs[j+1:]
        for rest in matchings(rem): out.append(tuple(sorted(((min(a,b),max(a,b)),)+rest)))
    return out

def choose(xs,k):
    import itertools
    return list(itertools.combinations(xs,k))

def diagrams():
    out=[]
    for r in OUTER_DEGREES:
      for s in range(6):
       for cross in range(min(r,s)+1):
        ro=r-cross; ri=s-cross
        if ro+ri>6: continue
        for om in choose(EXTERNALS,ro):
         rem=[x for x in EXTERNALS if x not in om]
         for im in choose(rem,ri):
          unmatched=[x for x in rem if x not in im]
          if len(unmatched)%2: continue
          for pairs in matchings(unmatched): out.append((r,s,cross,om,im,pairs))
    return sorted(out)

def pack(diagram):
    r,s,cross,om,im,pairs=diagram
    omask=sum(1<<x for x in om); imask=sum(1<<x for x in im)
    packed=0
    for n,(a,b) in enumerate(pairs): packed |= a<<(6*n); packed |= b<<(6*n+3)
    return DESCRIPTOR.pack(r,s,cross,omask,imask,len(pairs),0,packed,1,1)

def features(d):
    r,s,c,om,im,pairs=d
    f={('r',r),('s',s),('cross',c),('pairs',len(pairs))}
    f|={('outer_external',x) for x in om};f|={('inner_external',x) for x in im}
    f|={('metric_pair',a,b) for a,b in pairs}
    return f

def greedy_canary(ds):
    universe=set().union(*(features(d) for d in ds)); selected=[]; covered=set()
    while covered!=universe:
        ordinal=max(range(len(ds)),key=lambda i:(len(features(ds[i])-covered),-i) if i not in selected else (-1,0))
        selected.append(ordinal);covered|=features(ds[ordinal])
    return selected,sorted(universe,key=repr)

def main():
    ds=diagrams();assert len(ds)==400 and len(set(ds))==400
    blob=b''.join(map(pack,ds));assert len(blob)==6400
    selected,universe=greedy_canary(ds)
    source=ROOT/'src/eleven_dimensional_d21_invariant_diagrams.rs'
    raw_sha=hashlib.sha256(blob).hexdigest()
    semantic=hashlib.sha256(b'adynkra-11d-d21-device-diagram-v1\0'+blob).hexdigest()
    by_r=Counter(d[0] for d in ds);by_s=Counter(d[1] for d in ds)
    assert dict(sorted(by_r.items()))=={0:21,3:209,4:170}
    assert dict(sorted(by_s.items()))=={0:9,1:48,2:75,3:98,4:107,5:63}
    report={
      'schema_version':'adynkra-11d-d21-device-diagram-manifest-v1',
      'source_module_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'descriptor_layout':'<BBBBBBHIhH little endian: r,s,cross,outer_mask,inner_mask,pair_count,reserved,pairs_packed,normalization_numerator,normalization_denominator',
      'descriptor_bytes':16,'diagram_count':400,'blob_bytes':len(blob),
      'blob_sha256':raw_sha,'semantic_sha256':semantic,
      'diagrams_by_outer_degree':dict(sorted(by_r.items())),
      'diagrams_by_inner_degree':dict(sorted(by_s.items())),
      'normalization':'one canonical antisymmetric cross-contraction per internal-axis combination; no implicit cross-factorial',
      'external_index_codes':{'momentum':0,'h_vector':1,'output0':2,'output1':3,'output2':4,'output3':5},
      'pair_encoding':'three bits per endpoint, six bits per sorted pair, at most three lexicographically sorted pairs in low 18 bits',
      'validation':{'reserved_zero':True,'masks_disjoint':True,'pair_endpoints_are_exact_complement':True,'popcount_outer_equals_r_minus_cross':True,'popcount_inner_equals_s_minus_cross':True,'normalization_denominator_positive':True},
      'cpu_canary_greedy_ordinals':selected,
      'cpu_canary_diagram_count':len(selected),
      'cpu_canary_feature_count':len(universe),
      'cpu_canary_features':[repr(x) for x in universe],
      'passed_manifest':True,
      'boundary':'This freezes the current 400-signature metric/Clifford grammar. It does not by itself prove epsilon/Hodge completeness, Cartesian equivariance, or sector rank 52.'
    }
    out_blob=ROOT/'results/adynkra_11d_d21_device_diagrams_v1.bin'
    tmp=out_blob.with_suffix('.bin.tmp');tmp.write_bytes(blob);tmp.replace(out_blob)
    print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__':main()
