"""Regenerate data/tendim_10d_lr.json: the 10D N=1 supergravity L/R matrices.

Source: Cigliano, Dahl, Gates et al., "10D Supergravity Numerical Data Sets for
L & R Matrices", arXiv:2512.12157, and its data repository
github.com/mcmulaz/Super-Sym (the "Garden Algebra" Mathematica file, which ships
GENERATIVE code rather than literal matrices). This script is a Python
re-implementation of that construction (sigma-matrix Kronecker products + the
gauge-fixed H/B/Phi and psi/chi field ordering).

PROVENANCE CAVEAT: the output is a regeneration, not a literal download. It
satisfies the bosonic Garden relation L_I R_J + L_J R_I = 2 d_IJ I_82 to ~1e-12
with a nonzero fermionic remnant E_IJ; that is evidence of algebraic
plausibility, NOT proof of byte-fidelity to the authors' published matrices.
See REFERENCES.md and src/tendim_data.rs.
"""
import numpy as np
import json
import os
import sys

# ---- Output path resolution (no hardcoded absolute paths) ----
# Requirement: reproducible for any user / CI. Derive the default output path
# from this script's location (<repo>/data/tendim_10d_lr.json), and allow an
# argv[1] override so callers (e.g. CI) can redirect the artifact.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "tendim_10d_lr.json")
OUT_PATH = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

# ---- Numerical tolerances ----
# Entries are derived from sigma-matrix Kronecker products; the construction is
# real-valued up to floating-point round-off. Any imaginary part larger than
# this tolerance signals a transcription/algebra error, so we HARD-FAIL rather
# than silently drop Im(.) (see the assert before serialization below).
IMAG_TOL = 1e-9
# Fixed decimal count for canonical float serialization (see write block).
ROUND_DECIMALS = 12

# ---- Pauli / sigma setup (16x16 via Kronecker of 4) ----
s1 = np.array([[0,1],[1,0]], dtype=complex)
s2 = np.array([[0,-1j],[1j,0]], dtype=complex)
s3 = np.array([[1,0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

def kron4(a,b,c,d):
    return np.kron(np.kron(np.kron(a,b),c),d)

sigUp = {
    0: kron4(I2,I2,I2,I2),
    1: kron4(s2,s2,s2,s2),
    2: kron4(s2,s2,I2,s1),
    3: kron4(s2,s2,I2,s3),
    4: kron4(s2,s1,s2,I2),
    5: kron4(s2,s3,s2,I2),
    6: kron4(s2,I2,s1,s2),
    7: kron4(s2,I2,s3,s2),
    8: kron4(s1,I2,I2,I2),
    9: kron4(s3,I2,I2,I2),
}

def etasign(mu):
    return -1 if mu==0 else 1

def sigma(mu):       # \[Sigma][mu]
    return etasign(mu)*sigUp[mu]
def sigmaTildeUp(mu):
    return etasign(mu)*sigUp[mu]
def sigmaTilde(mu):  # SigmaTilde
    return etasign(mu)*sigmaTildeUp(mu)

def Sig2Up(mu,nu):
    return 0.5*(sigUp[mu].dot(sigmaTildeUp(nu)) - sigUp[nu].dot(sigmaTildeUp(mu)))
def Sig2(mu,nu):
    return etasign(mu)*etasign(nu)*Sig2Up(mu,nu)
def TildeSig2Up(mu,nu):
    return 0.5*(sigmaTildeUp(mu).dot(sigUp[nu]) - sigmaTildeUp(nu).dot(sigUp[mu]))

from itertools import permutations
def perm_sign(p):
    # signature of permutation p (list)
    n=len(p); s=1
    p=list(p)
    for i in range(n):
        for j in range(i+1,n):
            if p[i]>p[j]: s=-s
    return s

# PORT CAVEAT (Sig3Up / Signature parity): Mathematica's Sig3Up uses
# Signature[p] where p is a permutation of the *values* {mu,nu,rho}, i.e. the
# parity of the permutation needed to sort those values into canonical order.
# Here we instead iterate over permutations of the *positions* range(3) and use
# perm_sign of that positional permutation. These two parities coincide ONLY
# when the supplied (mu,nu,rho) are the distinct, already-ascending arguments
# that the current L/R construction actually passes (Sig3Up is always called as
# Sig3Up(0,rho,xi) with rho<xi, all distinct -> matches Signature of the sorted
# value list up to the same overall sign on every term, which cancels under the
# (1/6) symmetrization). It is NOT a faithful general port of Signature[p]: for
# repeated indices or non-ascending value arguments the two would diverge. Do
# not reuse Sig3Up outside the present call sites without revalidating.
def Sig3Up(mu,nu,rho):
    base=[mu,nu,rho]
    tot=np.zeros((16,16),dtype=complex)
    for p in permutations(range(3)):
        idx=[base[p[0]],base[p[1]],base[p[2]]]
        sg=perm_sign(p)  # positional parity; see PORT CAVEAT above
        tot += sg*(sigUp[idx[0]].dot(sigmaTildeUp(idx[1])).dot(sigUp[idx[2]]))
    return (1.0/6.0)*tot

def MixedLeft(mu,nu,rho,xi):
    return sigmaTilde(mu).dot(Sig3Up(nu,rho,xi))

# OMITTED MATHEMATICA FUNCTIONS (Garden Algebra source, lines 52-63):
#   - Sig3UpT[mu,nu,rho]      (the SigmaTilde . Sigma . SigmaTilde variant)
#   - MixedRight[nu,rho,xi,mu] (= Sig3UpT[nu,rho,xi] . Sigma[mu])
# These are intentionally NOT ported. The current L/R construction below builds
# its rows entirely from sigma/SigmaTilde, Sig2/TildeSig2Up, Sig3Up and
# MixedLeft; it never references Sig3UpT or MixedRight. Porting them would add
# dead code. Consequently this file is a port of the L/R construction actually
# used, NOT a complete line-for-line port of the entire Garden Algebra file.

# ---- Ordering / gauge fix ----
HPairsAll=[(mu,nu) for mu in range(0,10) for nu in range(mu,10)]   # 55
BPairsAll=[(mu,nu) for mu in range(0,10) for nu in range(mu+1,10)] # 45
HPairsGF=[p for p in HPairsAll if p[0]!=0]  # 45
BPairsGF=[p for p in BPairsAll if p[0]!=0]  # 36

HOrder=[(1,1)]+[p for p in HPairsGF if p!=(1,1)]
BOrder=list(BPairsGF)

HIndex={k:i+1 for i,k in enumerate(HOrder)}  # 1-based
BIndex={k:i+1 for i,k in enumerate(BOrder)}
betaScale=1
colPhi=len(HOrder)+len(BOrder)+1
assert colPhi==82, colPhi
assert len(HOrder)==45 and len(BOrder)==36

# row is 0-based numpy vector length colPhi
def addH(row,key,val):
    if key in HIndex:
        row[HIndex[key]-1]+=val
def addB(row,key,val):
    if key in BIndex:
        row[len(HOrder)+BIndex[key]-1]+=val

# ---- Fermion indexing (1-based in Mma) ----
spinorCount=16
def psiRow(mu,b):  # 1..160
    return mu*spinorCount+b
def chiRow(b):     # 161..176
    return 10*spinorCount+b
def colPsi(mu,b):
    return psiRow(mu,b)
def colChi(b):
    return chiRow(b)

# ---- R[a]: bosons->fermions, but built rowwise length colPhi; a is 1..16 ----
def rowForPsi(mu,dotb,a):  # dotb,a 1-based
    row=np.zeros(colPhi,dtype=complex)
    # term1: h
    for rho in range(0,10):
        coeff=-(0.5)*TildeSig2Up(0,rho)[dotb-1,a-1]
        keyH=tuple(sorted((rho,mu)))
        addH(row,keyH,coeff)
    # term2: B
    for rho in range(0,10):
        if rho!=mu:
            coeffB=-(0.5)*TildeSig2Up(0,rho)[dotb-1,a-1]
            if rho<mu:
                keyB=(rho,mu); signB=1
            else:
                keyB=(mu,rho); signB=-1
            addB(row,keyB,betaScale*signB*coeffB)
    # term3: A_[3]
    for rho in range(0,10):
        for xi in range(rho+1,10):
            keyB=(rho,xi)
            coeffA=(1.0/16.0)*MixedLeft(mu,0,rho,xi)[dotb-1,a-1]
            addB(row,keyB,betaScale*coeffA)
    return row

def rowForChi(b,a):
    row=np.zeros(colPhi,dtype=complex)
    row[colPhi-1]+=(1.0/np.sqrt(8))*sigUp[0][b-1,a-1]
    for rho in range(0,10):
        for xi in range(rho+1,10):
            keyB=(rho,xi)
            addB(row,keyB,betaScale*(-(1.0/8.0))*Sig3Up(0,rho,xi)[b-1,a-1])
    return row

def Rmat(a):
    psiBlock=[]
    for mu in range(0,10):
        for dotb in range(1,17):
            psiBlock.append(rowForPsi(mu,dotb,a))
    chiBlock=[rowForChi(b,a) for b in range(1,17)]
    M=np.array(psiBlock+chiBlock,dtype=complex)  # 176 x 82
    assert M.shape==(176,82), M.shape
    return M

# ---- L[a]: fermions->bosons (rows length 176) ----
def coeffRow(mu,a):       # sigma[mu][a,All] -> length16
    return sigma(mu)[a-1,:]
def coeffRow2(mu,nu,a):
    return Sig2(mu,nu)[a-1,:]

def rowForH(mu,nu,a):
    row=np.zeros(176,dtype=complex)
    cmu=coeffRow(mu,a); cnu=coeffRow(nu,a)
    for b in range(1,17):
        cm=cmu[b-1]; cn=cnu[b-1]
        if cm!=0: row[colPsi(nu,b)-1]+=cm
        if cn!=0: row[colPsi(mu,b)-1]+=cn
    return row

def rowForB(mu,nu,a):
    row=np.zeros(176,dtype=complex)
    cmu=coeffRow(mu,a); cnu=coeffRow(nu,a)
    c2=coeffRow2(mu,nu,a)
    for b in range(1,17):
        cm=cmu[b-1]; cn=cnu[b-1]
        if cm!=0: row[colPsi(nu,b)-1]+=cm
        if cn!=0: row[colPsi(mu,b)-1]+=-cn
    for b in range(1,17):
        coeff=c2[b-1]
        if coeff!=0: row[colChi(b)-1]+=coeff
    return betaScale*row

def PhiRowVector(a):
    row=np.zeros(176,dtype=complex)
    row[colChi(a)-1]=np.sqrt(8)
    return row

def Lmat(a):
    rows=[]
    for (mu,nu) in HOrder:
        rows.append(rowForH(mu,nu,a))
    for (mu,nu) in BOrder:
        rows.append(rowForB(mu,nu,a))
    rows.append(PhiRowVector(a))
    M=np.array(rows,dtype=complex)  # 82 x 176
    assert M.shape==(82,176), M.shape
    return M

# ---- Build all ----
Ls=[Lmat(a) for a in range(1,17)]
Rs=[Rmat(a) for a in range(1,17)]

# ---- Verify Garden relations ----
I82=np.eye(82); I176=np.eye(176)
max_bos=0.0
for i in range(16):
    for j in range(16):
        lhs=Ls[i].dot(Rs[j])+Ls[j].dot(Rs[i])
        rhs=2*(1 if i==j else 0)*I82
        err=np.linalg.norm(lhs-rhs)
        if err>max_bos: max_bos=err
print("Max bosonic residual (||L_i R_j + L_j R_i - 2 d_ij I_82||):", max_bos)

# Fermionic: R_i L_j + R_j L_i = 2 d_ij I + 2 E_ij ; E_ij = (RL+RL)/2 - d_ij I
max_imag=0.0
diag_e_norm=0.0
offdiag_e_norms=[]
for i in range(16):
    for j in range(16):
        lhs=Rs[i].dot(Ls[j])+Rs[j].dot(Ls[i])
        E=0.5*lhs-(1 if i==j else 0)*I176
        max_imag=max(max_imag, np.max(np.abs(E.imag)))
        if i==j:
            diag_e_norm=max(diag_e_norm,np.linalg.norm(E))
        elif i<j:
            offdiag_e_norms.append((i+1,j+1,np.linalg.norm(E)))
print("Max |Im(L,R entries-derived E)|:", max_imag)
print("Max diagonal E_ii Frobenius norm (should be ~0 if i==i closes):", diag_e_norm)
nz=[(i,j,n) for (i,j,n) in offdiag_e_norms if n>1e-9]
print("Number of off-diagonal (i<j) pairs with nonzero E_ij:", len(nz), "out of", len(offdiag_e_norms))
if nz:
    print("Sample E_ij Frobenius norms (i,j,norm):", nz[:5])
    print("Range of nonzero E_ij norms:", min(n for _,_,n in nz), "to", max(n for _,_,n in nz))

# Check if entries are real (imag negligible) for L,R themselves
maxImL=max(np.max(np.abs(L.imag)) for L in Ls)
maxImR=max(np.max(np.abs(R.imag)) for R in Rs)
print("Max |Im(L)|:", maxImL, " Max |Im(R)|:", maxImR)

# distinct entry values
allvals=set()
for L in Ls:
    for v in np.round(L.real.flatten(),6):
        allvals.add(v)
print("Distinct real entry values in L (sample):", sorted(allvals)[:20], "... total", len(allvals))

# ---- HARD-FAIL on imaginary parts (do NOT silently drop Im) ----
# The matrices must be real to floating-point precision. If any imaginary part
# exceeds IMAG_TOL the algebra/transcription is wrong and serializing the real
# part would hide it, so abort with a clear, actionable message.
max_im_L = max(np.max(np.abs(L.imag)) for L in Ls)
max_im_R = max(np.max(np.abs(R.imag)) for R in Rs)
assert max_im_L <= IMAG_TOL, (
    "Imaginary part of L exceeds tolerance: max|Im(L)|=%.3e > IMAG_TOL=%.1e. "
    "The construction should be real; refusing to drop a nonzero imaginary "
    "part. Check the sigma-matrix algebra / coefficients." % (max_im_L, IMAG_TOL)
)
assert max_im_R <= IMAG_TOL, (
    "Imaginary part of R exceeds tolerance: max|Im(R)|=%.3e > IMAG_TOL=%.1e. "
    "The construction should be real; refusing to drop a nonzero imaginary "
    "part. Check the sigma-matrix algebra / coefficients." % (max_im_R, IMAG_TOL)
)
print("Imaginary-part check passed (max|Im(L)|=%.3e, max|Im(R)|=%.3e <= %.1e)"
      % (max_im_L, max_im_R, IMAG_TOL))

# ---- Canonical, deterministic serialization ----
# Determinism requirements (so re-running yields BYTE-IDENTICAL output):
#   * Fixed float formatting: every entry rounded to ROUND_DECIMALS decimals,
#     then emitted by json with default repr (a -0.0 from rounding is normalized
#     to 0.0 below so sign-of-zero never flips between runs).
#   * Fixed separators: json defaults (", " and ": ") are explicit and stable.
#   * Fixed key order: keys are emitted in an explicit canonical order rather
#     than relying on dict insertion order. NOTE: we deliberately do NOT use
#     json sort_keys=True. The committed artifact predates this hardening and
#     was written in the order below; using a fixed explicit order keeps the
#     output deterministic AND byte-identical to the committed file. (Alphabetic
#     sort_keys would reorder L/R/n/... and change the bytes for no benefit.)
def to_list(M):
    out=[]
    for row in M.real:
        r=[]
        for x in row:
            v=round(float(x), ROUND_DECIMALS)
            if v==0.0:
                v=0.0  # normalize -0.0 -> 0.0 for stable bytes
            r.append(v)
        out.append(r)
    return out

# Explicit canonical key order (see note above). The "source" field is written
# here; the richer "provenance" block (research metadata: upstream commit/hashes,
# license posture, the resolved Eq 6.0.5 typo note) is maintained out-of-band in
# PROVENANCE.md and carried in the committed JSON. To avoid clobbering it on
# regeneration, we PRESERVE an existing provenance block verbatim (inserted in the
# same position, so regeneration stays byte-identical to the committed file). The
# reproducibility guarantee is the canonical_token_hash() of the MATRIX content
# (src/tendim_data.rs), which is independent of this metadata.
_existing_provenance = None
if os.path.exists(OUT_PATH):
    # Fail LOUDLY if the existing file is present but unreadable: silently dropping
    # the provenance block on a parse error would erase research metadata without
    # warning. A genuinely missing file (fresh generation) is the only case where
    # provenance is legitimately absent.
    with open(OUT_PATH) as _pf:
        _existing = json.load(_pf)  # raises on malformed JSON -> abort, do not clobber
    _existing_provenance = _existing.get("provenance")
    if _existing_provenance is None:
        print("WARNING: existing JSON has no 'provenance' block to preserve.")

data={
    "nb":82,"nf":176,"n":16,
    "source":"arXiv:2512.12157 / github.com/mcmulaz/Super-Sym 'Garden Algebra' Mathematica code, evaluated numerically",
    "relations":{
        "bosonic":"L_I R_J + L_J R_I = 2 delta_IJ I_82",
        "fermionic":"R_I L_J + R_J L_I = 2 delta_IJ I_176 + 2 E_IJ"
    },
}
if _existing_provenance is not None:
    data["provenance"]=_existing_provenance
data["L"]=[to_list(L) for L in Ls]
data["R"]=[to_list(R) for R in Rs]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(data, f, separators=(", ", ": "), ensure_ascii=True)
print("Wrote", OUT_PATH)
print("File size bytes:", os.path.getsize(OUT_PATH))
