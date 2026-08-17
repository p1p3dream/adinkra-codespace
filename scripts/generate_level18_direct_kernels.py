#!/usr/bin/env python3
"""Generate the four non-Hodge level-18 B5 highest-weight kernel systems.

The calculation is exact. It builds the five sparse Chevalley raising blocks,
performs deterministic sparse row echelon reduction over the Mersenne prime
2^31-1, rationally reconstructs the normalized nullspace, and accepts a
primitive integer vector only after direct integer verification against every
raising row. Completed labels are checkpointed independently.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import struct
import time
from fractions import Fraction
from pathlib import Path

PRIME = 2_147_483_647
ROOTS = ((2,-2,0,0,0),(0,2,-2,0,0),(0,0,2,-2,0),(0,0,0,2,-2),(0,0,0,0,2))
SPECS = {"12000":1,"11100":3,"11010":5,"11002":6}
WEIGHTS = tuple(tuple(1 if ((i>>(4-a))&1)==0 else -1 for a in range(5)) for i in range(32))
WEIGHT_INDEX = {weight:index for index,weight in enumerate(WEIGHTS)}
RAISE = tuple(tuple(WEIGHT_INDEX.get(tuple(WEIGHTS[i][a]+ROOTS[root][a] for a in range(5)),-1) for i in range(32)) for root in range(5))


def highest(label: str) -> tuple[int,...]:
    digits=tuple(map(int,label))
    return tuple(2*sum(digits[i:4])+digits[4] for i in range(5))


def half_groups(offset: int):
    groups={}
    half_weights=WEIGHTS[offset:offset+16]
    for mask in range(1<<16):
        weight=[0]*5
        remainder=mask
        while remainder:
            bit=(remainder&-remainder).bit_length()-1
            remainder&=remainder-1
            for axis in range(5): weight[axis]+=half_weights[bit][axis]
        groups.setdefault((mask.bit_count(),tuple(weight)),[]).append(mask)
    return groups


def weight_basis(degree,target,left,right):
    result=[]
    for left_degree in range(max(0,degree-16),min(16,degree)+1):
        right_degree=degree-left_degree
        for (candidate_degree,left_weight),left_masks in left.items():
            if candidate_degree!=left_degree: continue
            needed=tuple(target[a]-left_weight[a] for a in range(5))
            right_masks=right.get((right_degree,needed))
            if right_masks:
                result.extend(x|(y<<16) for x in left_masks for y in right_masks)
    result.sort()
    return result


def raising_rows(label,left,right):
    target=highest(label)
    source=weight_basis(18,target,left,right)
    blocks=[]
    row_count=0
    for root in range(5):
        output_weight=tuple(target[a]+ROOTS[root][a] for a in range(5))
        basis=weight_basis(18,output_weight,left,right)
        blocks.append({mask:index+row_count for index,mask in enumerate(basis)})
        row_count+=len(basis)
    rows=[{} for _ in range(row_count)]
    for column,mask in enumerate(source):
        for root in range(5):
            for lower in range(32):
                upper=RAISE[root][lower]
                if upper<0 or not(mask>>lower&1) or mask>>upper&1: continue
                output=(mask^(1<<lower))|(1<<upper)
                low,high=sorted((lower,upper))
                interval=0 if high==low+1 else ((1<<high)-1)^((1<<(low+1))-1)
                sign=-1 if (mask&interval).bit_count()%2 else 1
                rows[blocks[root][output]][column]=sign
    return source,rows


def sparse_echelon(rows,columns,prime):
    pivots={}
    maximum_pivot_width=0
    for row in sorted(rows,key=len):
        row={column:value%prime for column,value in row.items()}
        while row:
            column=min(row)
            coefficient=row[column]
            pivot=pivots.get(column)
            if pivot is None:
                inverse=pow(coefficient,-1,prime)
                row={key:(value*inverse)%prime for key,value in row.items() if (value*inverse)%prime}
                pivots[column]=row
                maximum_pivot_width=max(maximum_pivot_width,len(row))
                break
            for key,value in pivot.items():
                reduced=(row.get(key,0)-coefficient*value)%prime
                if reduced: row[key]=reduced
                else: row.pop(key,None)
    return pivots,maximum_pivot_width


def rational_reconstruct(residue,modulus):
    residue%=modulus
    bound=math.isqrt(modulus//2)
    old_remainder,remainder=modulus,residue
    old_denominator,denominator=0,1
    while abs(remainder)>bound:
        quotient=old_remainder//remainder
        old_remainder,remainder=remainder,old_remainder-quotient*remainder
        old_denominator,denominator=denominator,old_denominator-quotient*denominator
    if denominator==0: return None
    if denominator<0: remainder,denominator=-remainder,-denominator
    divisor=math.gcd(abs(remainder),denominator)
    numerator=remainder//divisor
    denominator//=divisor
    if abs(numerator)<=bound and denominator<=bound and (residue*denominator-numerator)%modulus==0:
        return Fraction(numerator,denominator)
    return None


def primitive_integer_nullspace(rows,columns,pivots,prime):
    free=[column for column in range(columns) if column not in pivots]
    result=[]
    for free_column in free:
        modular={free_column:1}
        for column in sorted(pivots,reverse=True):
            row=pivots[column]
            value=-sum(coefficient*modular.get(other,0) for other,coefficient in row.items() if other!=column)%prime
            if value: modular[column]=value
        rationals=[]
        for column in range(columns):
            value=rational_reconstruct(modular.get(column,0),prime)
            if value is None:
                raise RuntimeError(f"rational reconstruction failed at column {column}")
            rationals.append(value)
        denominator=1
        for value in rationals: denominator=math.lcm(denominator,value.denominator)
        vector=[value.numerator*(denominator//value.denominator) for value in rationals]
        divisor=0
        for value in vector: divisor=math.gcd(divisor,abs(value))
        vector=[value//divisor for value in vector]
        first=next(value for value in vector if value)
        if first<0: vector=[-value for value in vector]
        for row in rows:
            residual=sum(coefficient*vector[column] for column,coefficient in row.items())
            if residual:
                raise RuntimeError(f"nonzero exact raising residual {residual}")
        result.append(vector)
    return free,result


def write_label(label,copies,left,right,root):
    started=time.time()
    source,rows=raising_rows(label,left,right)
    built=time.time()
    pivots,maximum_pivot_width=sparse_echelon(rows,len(source),PRIME)
    reduced=time.time()
    free,vectors=primitive_integer_nullspace(rows,len(source),pivots,PRIME)
    verified=time.time()
    if len(vectors)!=copies:
        raise RuntimeError(f"{label}: expected nullity {copies}, found {len(vectors)}")
    maximum=max(abs(value) for vector in vectors for value in vector)
    width=2 if maximum<=32767 else 4
    kernel_dir=root/"data/eleven_dimensional_spinor_bridge"
    kernel_dir.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for copy,vector in enumerate(vectors,1):
        suffix="" if copies==1 else f"_{copy}"
        path=kernel_dir/f"level18_{label}_highest_weight_kernel{suffix}.i{8*width}le"
        temporary=path.with_suffix(path.suffix+f".{__import__('os').getpid()}.tmp")
        with temporary.open("wb") as stream:
            for value in vector: stream.write(struct.pack("<h" if width==2 else "<i",value))
        temporary.replace(path)
        outputs.append({"copy":copy,"path":str(path.relative_to(root)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size,"nonzero_coefficients":sum(value!=0 for value in vector),"maximum_absolute_coefficient":max(map(abs,vector))})
    return {"dynkin_label":label,"exterior_degree":18,"source_columns":len(source),"raising_rows":len(rows),"nonzero_entries":sum(map(len,rows)),"prime":PRIME,"exact_modular_rank":len(pivots),"exact_nullity":len(vectors),"free_columns":free,"maximum_pivot_width":maximum_pivot_width,"coefficient_width_bytes":width,"outputs":outputs,"seconds":{"matrix":built-started,"echelon":reduced-built,"reconstruct_and_integer_verify":verified-reduced,"total":verified-started},"passed":True}


def empty_artifact():
    return {"schema_version":"adynkra-11d-level18-direct-kernel-generation-v1","method":"deterministic exact sparse echelon over 2^31-1, rational reconstruction, and full integer residual verification","systems":[]}


def verify_outputs(artifact,root):
    for system in artifact.get("systems",[]):
        if system["exact_modular_rank"]+system["exact_nullity"]!=system["source_columns"]:
            raise RuntimeError(f"rank-nullity mismatch for {system['dynkin_label']}")
        for output in system["outputs"]:
            path=root/output["path"]
            if not path.exists() or path.stat().st_size!=output["bytes"]:
                raise RuntimeError(f"missing or truncated kernel {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=output["sha256"]:
                raise RuntimeError(f"kernel hash mismatch {path}")


def publish_checkpoint(checkpoint,root,new_systems):
    lock_path=checkpoint.with_suffix(checkpoint.suffix+".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        artifact=json.loads(checkpoint.read_text()) if checkpoint.exists() else empty_artifact()
        by_label={system["dynkin_label"]:system for system in artifact.get("systems",[])}
        for system in new_systems:
            existing=by_label.get(system["dynkin_label"])
            if existing is not None:
                old_hashes=[output["sha256"] for output in existing["outputs"]]
                new_hashes=[output["sha256"] for output in system["outputs"]]
                if old_hashes!=new_hashes:
                    raise RuntimeError(f"conflicting exact kernels for {system['dynkin_label']}")
            by_label[system["dynkin_label"]]=system
        artifact["systems"]=sorted(by_label.values(),key=lambda item:tuple(SPECS).index(item["dynkin_label"]))
        artifact["completed_systems"]=len(artifact["systems"])
        artifact["completed_kernel_copies"]=sum(item["exact_nullity"] for item in artifact["systems"])
        artifact["passed"]=artifact["completed_systems"]==4 and artifact["completed_kernel_copies"]==15
        verify_outputs(artifact,root)
        temporary=checkpoint.with_suffix(checkpoint.suffix+f".{__import__('os').getpid()}.tmp")
        temporary.write_text(json.dumps(artifact,indent=2)+"\n")
        temporary.replace(checkpoint)
        fcntl.flock(lock,fcntl.LOCK_UN)
    return artifact


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("labels",nargs="*",choices=tuple(SPECS),default=tuple(SPECS))
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force",action="store_true")
    parser.add_argument("--merge-checkpoints",type=Path,nargs="*",default=[])
    args=parser.parse_args()
    root=args.root.resolve()
    checkpoint=root/"results/adynkra_11d_level18_direct_kernel_generation.json"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if args.merge_checkpoints:
        systems=[]
        for path in args.merge_checkpoints:
            systems.extend(json.loads(path.read_text())["systems"])
        artifact=publish_checkpoint(checkpoint,root,systems)
        print(f"merged {artifact['completed_systems']} exact systems and {artifact['completed_kernel_copies']} kernels",flush=True)
        return
    artifact=json.loads(checkpoint.read_text()) if checkpoint.exists() and not args.force else empty_artifact()
    completed={system["dynkin_label"] for system in artifact["systems"] if system.get("passed")}
    left,right=half_groups(0),half_groups(16)
    for label in args.labels:
        if label in completed and not args.force:
            print(f"{label}: checkpoint complete",flush=True); continue
        print(f"{label}: building",flush=True)
        system=write_label(label,SPECS[label],left,right,root)
        artifact=publish_checkpoint(checkpoint,root,[system])
        print(f"{label}: exact nullity {system['exact_nullity']} in {system['seconds']['total']:.2f}s",flush=True)

if __name__=="__main__": main()
