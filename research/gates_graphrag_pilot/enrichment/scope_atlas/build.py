#!/usr/bin/env python3
"""Build evidence-backed scope and atlas proposals from the pilot chunk corpus."""
import json, re
from pathlib import Path

HERE=Path(__file__).resolve().parent
CHUNKS=Path('/tmp/gates-graphrag-pilot/chunks-enriched.jsonl')
OUT=HERE/'proposals.jsonl'

def norm(s): return re.sub(r'\s+',' ',s).strip()
chunks={}
for line in CHUNKS.open(encoding='utf-8'):
    c=json.loads(line); chunks[c['chunk_id']]=c

def ent(t,k,n): return {'type':t,'key':f'{t}:{k}','name':n}
def paper(pid):
    c=next(c for c in chunks.values() if c['paper_id']==pid)
    return ent('paper',pid,c['title']) | {'key':f'arxiv:{pid}'}

spec=[]
def add(pid,rel,target,chunk_id,excerpt,source=None,confidence=.97,notes=None):
    c=chunks[chunk_id]
    ex=norm(excerpt)
    assert pid==c['paper_id'], (pid,chunk_id)
    assert ex in norm(c['text']), (chunk_id,ex)
    spec.append({
      'source': source or paper(pid), 'relationship':rel, 'target':target,
      'evidence':{'paper_id':pid,'chunk_id':chunk_id,'page_number':c['page_number'],
                  'section':c.get('section_heading'),'excerpt':ex},
      'basis':'explicit_text','review_status':'pending','confidence':confidence,
      **({'notes':notes} if notes else {})})

# 1911.00807
p='1911.00807'; c=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','10d-unconstrained-scalar-superfields','10D unconstrained scalar superfields'),c,
    'The ﬁrst complete and explicit SO(1,9) Lorentz descriptions of all com- ponent ﬁelds contained in the N = 1, N = 2A, and N = 2B unconstrained scalar 10D superﬁelds are presented.',notes='Scope properties: spacetime_dimension=10; supersymmetry=N=1, N=2A, N=2B.')
add(p,'USES_GROUP',ent('group','so-1-9','SO(1,9) Lorentz group'),c,
    'The ﬁrst complete and explicit SO(1,9) Lorentz descriptions of all com- ponent ﬁelds contained in the N = 1, N = 2A, and N = 2B unconstrained scalar 10D superﬁelds are presented.')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','so-1-9-spin-bundle-representation','spin-bundle representation of SO(1,9)'),c,
    'Adinkra graphs for ten dimensional superspaces are deﬁned for the ﬁrst time, whose nodes depict spin bundle representations of SO(1,9).')
add(p,'DESCRIBES_MULTIPLET',ent('multiplet','10d-off-shell-nordstrom-supergravity','off-shell 10D Nordström supergravity multiplet'),c,
    'A consequential deliverable of this advance is it provides the ﬁrst explicit, in terms of component ﬁelds, examples of all the oﬀ-shell 10D Nordstr¨om SG theories relevant to string theory, without oﬀ-shell central charges that are reducible but with ﬁnite numbers of ﬁelds.',notes='The paper describes reducible, finite-field examples and does not claim an irreducible formulation.')
scan=ent('method','breitenlohner-prepotential-scan','Breitenlohner-style prepotential scan')
add(p,'HAS_INPUT',ent('property','graviton-gravitino-component-content','graviton and gravitino component content'),c,
    'An analogue of Breitenlohner’s approach is implemented to scan for superﬁelds that contain graviton(s) and gravitino(s), which are the candidates for the superconformal prepotential superﬁelds of 10D oﬀ-shell supergravity theories and Yang-Mills theories.',source=scan)
add(p,'HAS_OUTPUT',ent('scope','10d-superconformal-prepotential-candidates','candidate 10D superconformal prepotential superfields'),c,
    'An analogue of Breitenlohner’s approach is implemented to scan for superﬁelds that contain graviton(s) and gravitino(s), which are the candidates for the superconformal prepotential superﬁelds of 10D oﬀ-shell supergravity theories and Yang-Mills theories.',source=scan,notes='The output is a candidate set, not a proof that the candidates are prepotentials.')

# 2002.08502
p='2002.08502'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','11d-n1-unconstrained-scalar-superfield','11D N = 1 unconstrained scalar superfield'),a,
    'For the ﬁrst time in the physics literature, the Lorentz representations of all 2,147,483,648 bosonic degrees of freedom and 2,147,483,648 fermionic degrees of freedom in an unconstrained eleven dimensional scalar superﬁeld are presented.',notes='Scope properties: spacetime_dimension=11; supersymmetry=N=1.')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','11d-scalar-superfield-lorentz-content','Lorentz-representation content of the 11D scalar superfield'),a,
    'For the ﬁrst time in the physics literature, the Lorentz representations of all 2,147,483,648 bosonic degrees of freedom and 2,147,483,648 fermionic degrees of freedom in an unconstrained eleven dimensional scalar superﬁeld are presented.')
add(p,'USES_GROUP',ent('group','so-1-10','SO(1,10) Lorentz group'),f'{p}:p0025:c001',
    'We can decompose θ-monomials θα1 · · · θαn into a direct sum of irreducible representations of Lorentz group SO(1,10).')
branch=ent('algorithm','projection-matrix-branching-algorithm','projection-matrix branching algorithm')
add(p,'HAS_INPUT',ent('mathematical_object','lie-algebra-projection-matrix','Lie-algebra projection matrix'),f'{p}:p0026:c001',
    'Thus, the algorithm for calculating a branching rule of an irrep R of g given the projection matrix can be summarized as follows.',source=branch)
add(p,'HAS_OUTPUT',ent('representation','projected-lie-subalgebra-irreps','irreducible representations of the projected Lie subalgebra'),f'{p}:p0026:c001',
    'Calculate the projected weight vector in h by Equation (4.6) for every weight vector in the weight diagram of R and get the projected weight diagram; 3. Find irrep(s) in h corresponding to the projected weight diagram.',source=branch)
add(p,'DESCRIBES_REPRESENTATION',ent('representation','so-1-10-conformal-graviton','SO(1,10) conformal-graviton representation'),a,
    'It is noted at level sixteen in the 11D, N = 1 scalar superﬁeld, the {65} representation of SO(1,10), the conformal graviton, is present.',notes='Scope property: superfield_level=16.')

# 2006.03609
p='2006.03609'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','10d-superspaces','10D superspaces'),a,
    'We explicitly discuss the cases of ten dimensional superspaces.',notes='Scope property: spacetime_dimension=10.')
add(p,'APPLIES_TO',ent('scope','supermultiplets-in-all-superspaces','component fields in supermultiplets across superspaces'),a,
    'These suggest a computation- ally direct way to describe the component ﬁelds contained within supermul- tiplets in all superspaces.',confidence=.84,notes='The paper presents this as a suggested general scope; the abstract says the explicit discussion is ten-dimensional.')
add(p,'DESCRIBES_MULTIPLET',ent('multiplet','10d-n1-scalar-superfield','10D N = 1 scalar-superfield supermultiplet'),a,
    'Advening to Adynkraﬁelds: Young Tableaux to Component Fields of the 10D, N = 1 Scalar Superﬁeld',notes='Scope properties: spacetime_dimension=10; supersymmetry=N=1.')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','component-field-dynkin-label','component-field representation encoded by a Dynkin label'),a,
    'We show this is possible by replacing conventional θ-expansions by expansions over Young Tableaux and component ﬁelds by Dynkin Labels.')
add(p,'USES_ALGEBRA',ent('algebra','garden-algebra','Garden algebra GR(d,N)'),f'{p}:p0064:c001',
    'The algebra for the L-matrices and R-matrices in (8.26) deﬁnes the GR(d, N) algebra or the “Garden Algebra” (d, N). In the present context d = 32,768 and N = 16.',notes='Scope properties: d=32768; color_or_supercharge_count=16.')
adf=ent('construction','adynkrafield-expansion','Adynkrafield expansion')
add(p,'HAS_INPUT',ent('mathematical_object','young-tableau','Young tableau'),a,
    'We show this is possible by replacing conventional θ-expansions by expansions over Young Tableaux and component ﬁelds by Dynkin Labels.',source=adf)
add(p,'HAS_INPUT',ent('mathematical_object','dynkin-label','Dynkin label'),a,
    'We show this is possible by replacing conventional θ-expansions by expansions over Young Tableaux and component ﬁelds by Dynkin Labels.',source=adf)
add(p,'HAS_OUTPUT',ent('construction','component-field-index-structure','component-field index structure'),a,
    'Without the need to introduce σ-matrices, this permits rapid passages from Adynkras →Young Tableaux →Component Field Index Structures for both bosonic and fermionic ﬁelds',source=adf)

# 2007.07390
p='2007.07390'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','lorentzian-spacetimes-d4-d9','Lorentzian spacetimes in dimensions four through nine'),a,
    'We present Adynkra Libraries that can be used to explore the embedding of multiplets of component ﬁeld (whether on-shell or partial on-shell) within Salam-Strathdee superﬁelds for theories in dimension nine through four.',notes='Scope properties: spacetime_dimension=4..9.')
add(p,'DESCRIBES_MULTIPLET',ent('multiplet','on-shell-or-partial-on-shell-component-field-multiplet','on-shell or partially on-shell component-field multiplet'),a,
    'We present Adynkra Libraries that can be used to explore the embedding of multiplets of component ﬁeld (whether on-shell or partial on-shell) within Salam-Strathdee superﬁelds for theories in dimension nine through four.')
add(p,'USES_GROUP',ent('group','so-1-d-minus-1','SO(1,D-1) Lorentz group'),f'{p}:p0006:c001',
    'We use the Minkowski signature (−, +, +, . . . , +) in every dimension, and the corresponding Lorentz group is SO(1, D−1).')
add(p,'USES_ALGEBRA',ent('algebra','su-d-to-so-1-d-minus-1-branching','su(d) to so(1,D-1) branching'),f'{p}:p0006:c002',
    'su(d) ⊃so(1, D −1) of the totally antisymmetric irreps in su(d) constructed by the fundamental representation {d}. These calculations can be accomplished by either Susyno [10] or LieART [11].')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','9d-minimal-scalar-superfield-dynkin-label-content','Dynkin-label representation content of the 9D minimal scalar superfield'),f'{p}:p0007:c002',
    'The 9D minimal superﬁeld component decomposition results by Dynkin Labels are shown below.',notes='Scope property: spacetime_dimension=9. The paper presents corresponding sections for dimensions eight through four as well.')
ada=ent('method','adynkra-digital-analysis-scan','Adynkra digital analysis scan')
add(p,'HAS_INPUT',ent('dataset','on-shell-component-field-spectrum','specified on-shell component-field spectrum'),f'{p}:p0055:c001',
    'One can now start with any spectrum of on-shell component ﬁelds in these various dimensions and algorithmically derive the minimal dimension superﬁeld representation (as well as alternatives) that contains this speciﬁed component ﬁeld spectrum.',source=ada)
add(p,'HAS_OUTPUT',ent('representation','minimal-superfield-containing-spectrum','minimal-dimension superfield representation containing the specified spectrum'),f'{p}:p0055:c001',
    'One can now start with any spectrum of on-shell component ﬁelds in these various dimensions and algorithmically derive the minimal dimension superﬁeld representation (as well as alternatives) that contains this speciﬁed component ﬁeld spectrum.',source=ada,notes='The passage also permits alternative containing representations.')
add(p,'HAS_OUTPUT',ent('dataset','component-multiplet-embedding-d4-d9','component-multiplet embeddings in dimensions four through nine'),a,
    'We present Adynkra Libraries that can be used to explore the embedding of multiplets of component ﬁeld (whether on-shell or partial on-shell) within Salam-Strathdee superﬁelds for theories in dimension nine through four.',source=ent('artifact','adynkra-library-d4-d9','Adynkra libraries for dimensions four through nine'),notes='The library supports exploration of embeddings; the edge does not assert that every possible embedding has been computed.')

# 2012.13308
p='2012.13308'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','4d-n-extended-susy-weight-space','weight space for 4D N-extended supersymmetry representations'),a,
    'A conjecture is made that the weight space for 4D, N-extended supersym- metrical representations is embedded within the permutahedra associated with permutation groups Sd.',confidence=.9,notes='The relationship records the paper’s conjectural scope, not an established embedding.')
add(p,'USES_GROUP',ent('group','s-d','symmetric group S_d'),a,
    'A conjecture is made that the weight space for 4D, N-extended supersym- metrical representations is embedded within the permutahedra associated with permutation groups Sd.',notes='The group degree d is a parameter, not a graph edge.')
add(p,'USES_GROUP',ent('group','coxeter-bc4','Coxeter group BC4'),f'{p}:p0032:c001',
    'In this work, we have used structures that are intrinsic to the Coxeter Group BC4 in order to discuss how their elements are organized to provide representations of four-dimensional N = 1 SUSY.',notes='Scope properties: spacetime_dimension=4; supersymmetry=N=1; color_count=4.')
add(p,'USES_GROUP',ent('group','s4','symmetric group S4'),f'{p}:p0003:c000',
    'for a ﬁxed value of bI, all of the matrices P bI, take the forms of elements in the permutation group (S4) of order four.',notes='Scope properties: permutation_degree=4; permutahedron_vertex_count=24.')
add(p,'USES_ALGEBRA',ent('algebra','garden-algebra','Garden algebra'),f'{p}:p0002:c001',
    'These real matrices satisfy an algebra given by LI RJ + LJ RI = 2 δI J I4 , RI LJ + RJ LI = 2 δI J I4 . (1.1) and which is referred to as “the Garden Algebra.”')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','minimal-off-shell-4d-n1-susy','minimal off-shell 4D N = 1 supersymmetry representation'),a,
    'Adinkras and Coxeter Groups associated with minimal representations of 4D, N = 1 supersymmetry provide evidence supporting this conjecture.')
add(p,'HAS_OUTPUT',ent('dataset','96-garden-algebra-quartets','96 quartets satisfying the Garden-algebra condition'),f'{p}:p0003:c000',
    'Thus, the matrices found by the search algorithm constitute 96 quartets of matrices that satisfy the condition shown in Eq. (1.3).',source=ent('algorithm','garden-algebra-matrix-search','Garden-algebra matrix search'))
add(p,'APPLIES_TO',ent('problem','off-shell-susy-auxiliary-field-problem','off-shell supersymmetry auxiliary-field problem'),a,
    'This observation suggest an en- tirely new way to approach the oﬀ-shell SUSY auxiliary ﬁeld problem based on IT algorithms probing the properties of Sd.',confidence=.8,notes='The abstract proposes a possible approach; it does not report a solution.')

# 2012.14015
p='2012.14015'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','4d-n1-pairs-forming-4d-n2','combinations of off-shell 4D N = 1 supermultiplets forming off-shell 4D N = 2 supermultiplets'),a,
    'We continue the search for rules that govern when oﬀ-shell 4D, N = 1 supermultiplets can be combined to form oﬀ-shell 4D, N = 2 supermultiplets.',notes='Scope properties: spacetime_dimension=4; input_supersymmetry=N=1; output_supersymmetry=N=2; input_color_count=4; output_color_count=8 after reduction.')
add(p,'USES_GROUP',ent('group','s8','symmetric group S8'),a,
    'We study the S8 permutations and Height Yielding Matrix Numbers (HYMN) embedded within the adinkras that correspond to these putative oﬀ-shell 4D, N = 2 supermultiplets.',notes='Scope properties: permutation_degree=8; permutahedron_vertex_count=40320.')
add(p,'USES_ALGEBRA',ent('algebra','garden-algebra','Garden algebra'),f'{p}:p0002:c001',
    'Every oﬀ-shell 4D, N = 1 supermultiplet reduced to one dimension leads to a set of matrices that satisfy the Garden algebra [16]')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','gr-8-8','GR(8,8) representation'),f'{p}:p0008:c001',
    'We believe this current work shows how the GR(8, 8) representation clearly impacts how oﬀ-shell 4D, N = 1 theories can be combined to become oﬀ-shell 4D, N = 2 theories.',confidence=.9,notes='The passage states the authors’ interpretation.')
for key,name in [('4d-n2-chiral-vector','off-shell 4D N = 2 chiral-vector combination'),('4d-n2-chiral-tensor','off-shell 4D N = 2 chiral-tensor combination')]:
    add(p,'DESCRIBES_MULTIPLET',ent('multiplet',key,name),a,
        'Only the combinations of the chiral + vector and chiral + tensor are found to have valises in the same class. This is consistent with the well known structure of 4D, N = 2 supermultiplets.')
hymn=ent('method','height-yielding-matrix-number','Height Yielding Matrix Number classification')
add(p,'HAS_INPUT',ent('construction','4d-n1-pair-forming-4d-n2-supermultiplet','pair of 4D N = 1 supermultiplets proposed to form a 4D N = 2 supermultiplet'),a,
    'Even though the HYMN deﬁnition was designed to distinguish between the raising and lowering of nodes in one dimensional valise supermultiplets, they are shown to accurately select out which combinations of oﬀ-shell 4D, N = 1 supermultiplets correspond to oﬀ-shell 4D, N = 2 super- multiplets.',source=hymn)
add(p,'HAS_OUTPUT',ent('result','selected-off-shell-4d-n2-combinations','selected off-shell 4D N = 2 supermultiplet combinations'),a,
    'Even though the HYMN deﬁnition was designed to distinguish between the raising and lowering of nodes in one dimensional valise supermultiplets, they are shown to accurately select out which combinations of oﬀ-shell 4D, N = 1 supermultiplets correspond to oﬀ-shell 4D, N = 2 super- multiplets.',source=hymn)

# 2304.09830
p='2304.09830'; a=f'{p}:p0001:c000'
alg=ent('algorithm','recursive-supermultiplet-construction','recursive supermultiplet construction')
add(p,'APPLIES_TO',ent('scope','arbitrary-n-extended-supermultiplets','arbitrary N-extended supermultiplet matrix collections'),a,
    'We study algorithms for recursively creating arbitrary N-extended ‘super- multiplets’ given minimal matrix representations of oﬀ-shell, N = 1 supermul- tiplet matrices.',notes='The paper defines “supermultiplet” operationally as an L/R matrix collection; N is a parameter.')
add(p,'HAS_INPUT',ent('representation','minimal-off-shell-n1-supermultiplet-matrices','minimal matrix representation of an off-shell N = 1 supermultiplet'),a,
    'We study algorithms for recursively creating arbitrary N-extended ‘super- multiplets’ given minimal matrix representations of oﬀ-shell, N = 1 supermul- tiplet matrices.',source=alg)
add(p,'HAS_OUTPUT',ent('representation','arbitrary-n-extended-supermultiplet-matrices','arbitrary N-extended supermultiplet matrix collection'),a,
    'We study algorithms for recursively creating arbitrary N-extended ‘super- multiplets’ given minimal matrix representations of oﬀ-shell, N = 1 supermul- tiplet matrices.',source=alg)
add(p,'USES_GROUP',ent('group','s8','symmetric group S8'),f'{p}:p0011:c000',
    'The absolute values of all seven sets lie within the permutation group of order eight.',notes='Scope properties: permutation_degree=8; permutahedron_vertex_count=40320.')
add(p,'USES_GROUP',ent('group','coxeter-group','Coxeter group'),f'{p}:p0009:c001',
    'The elements of the garden algebra, being a subgroup of the Coxeter group, are composed of a signed matrix (the boolean factors in the above equations) times a permutation matrix, which together form the Coxeter group elements.')
add(p,'USES_ALGEBRA',ent('algebra','garden-algebra','Garden algebra GR(d,N)'),f'{p}:p0007:c001',
    'In previous works, these have been called the GR(d, N) or “garden algebra” conditions recognizing the dependence on the parameters d and N.')
add(p,'USES_ALGEBRA',ent('algebra','euclidean-clifford-algebra','Euclidean Clifford algebra'),f'{p}:p0007:c001',
    'When the collections satisfy (and thus form representations of Euclidean Cliﬀord Algebras)')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','off-shell-lr-matrix-set','off-shell L/R matrix representation'),f'{p}:p0007:c001',
    'When the collections satisfy (and thus form representations of Euclidean Cliﬀord Algebras) { bγI , bγJ } = 2 δIJ I2d , (2.2) they are called an “oﬀ-shell” set.')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','on-shell-lr-matrix-set','on-shell L/R matrix representation'),f'{p}:p0007:c001',
    'In this case above, the collection is called an “on-shell” set. The coeﬃcients N IJbα (R) may be interpreted as obstructions of an oﬀ-shell set to forming a Cliﬀord algebra.')
for key,name in [('4d-n1-chiral','4D N = 1 chiral supermultiplet'),('4d-n1-vector','4D N = 1 vector supermultiplet'),('4d-n1-tensor','4D N = 1 tensor supermultiplet')]:
    add(p,'DESCRIBES_MULTIPLET',ent('multiplet',key,name),f'{p}:p0011:c000',
        'Starting from the 4D, N = 1 oﬀ-shell chiral, vector, and tensor supermultiplets, this leads to six sets of L-matrices.')
add(p,'APPLIES_TO',ent('scope','eight-supercharge-permutahedron','permutahedron setting for models with eight independent supercharges'),f'{p}:p0026:c001',
    'In this work, we have begun the exploration of the permutahedron associated with the omni- truncated 7-simplex. With regard to supersymmetry, this is the appropriate setting for models where eight independent supercharges occur.',notes='Scope properties: supercharge_count=8; polytope_vertex_count=40320.')

# 2311.06842
p='2311.06842'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','carroll-limit-supersymmetric-qft','Carroll limit of supersymmetric quantum field theory'),a,
    'Adinkra networks arise in the Carroll limit of supersymmetric QFT.')
add(p,'APPLIES_TO',ent('scope','familiar-4d-n1-supermultiplets','familiar 4D N = 1 supermultiplets'),a,
    'We call these “infinite unfolded” adinkras and study the properties of their realization on familiar 4D, N = 1 supermultiplets.',notes='Scope properties: spacetime_dimension=4; supersymmetry=N=1; color_count=4.')
add(p,'USES_ALGEBRA',ent('algebra','modified-garden-algebra','modified Garden algebra for unfolded Adinkras'),f'{p}:p0042:c001',
    'Compared to the folded adinkra construction process, we can also verify that the modified garden algebra holds for the unfolded field definitions.')
add(p,'USES_GROUP',ent('group','u3','U(3) Lie algebra'),f'{p}:p0038:c001',
    'The L-matrices (Eqs. (4.4) - (4.7)) and R-matrices (Eqs. (4.9) - (4.12)) are block diagonal which implies that they can be written in forms that utilize the diagonal generators of the u(3) Lie algebra in an outer product with 4 × 4 L-matrices and R-matrices.')
for key,name,chunk,excerpt in [
 ('4d-n1-chiral','4D N = 1 chiral supermultiplet',f'{p}:p0007:c001','The 4D, N = 1 chiral multiplet contains a scalar A, a pseudoscalar B, a Majorana fermion ψa, a scalar auxiliary field F, and a pseudoscalar auxiliary field G.'),
 ('4d-n1-vector','4D N = 1 vector supermultiplet',f'{p}:p0010:c001','The 4D, N = 1 vector multiplet is described by a vector Aµ, a Majorana fermion λa, and a pseudoscalar auxiliary field d.'),
 ('4d-n1-tensor','4D N = 1 tensor supermultiplet',f'{p}:p0013:c001','The 4D, N = 1 tensor multiplet consists of a scalar φ, a second-rank skew-symmetric tensor Bµ ν, and a Majorana fermion χa.'),
 ('complex-linear','complex linear supermultiplet',f'{p}:p0015:c001','The complex linear supermultiplet (CLS) contains scalar K, pseudoscalar L, Majorana spinor ζa and auxiliary scalar M, auxiliary pseudoscalar N, auxiliary vector Vµ, auxiliary axial-vector Uµ, and auxiliary Majorana spinors ρa and βa.')]:
    add(p,'DESCRIBES_MULTIPLET',ent('multiplet',key,name),chunk,excerpt)
red=ent('method','zero-brane-graph-reduction','0-brane graph reduction')
add(p,'HAS_INPUT',ent('dataset','4d-n1-chiral-component-fields','4D N = 1 chiral-multiplet component fields and transformation rules'),f'{p}:p0007:c001',
    'Based on Eq. (2.2) we can apply a 0-brane reduction process [16] for each field and draw the relations between each field as a graph, as seen in Fig. 6.',source=red)
add(p,'HAS_OUTPUT',ent('construction','reduced-component-field-graph','graph of relations among reduced component fields'),f'{p}:p0007:c001',
    'Based on Eq. (2.2) we can apply a 0-brane reduction process [16] for each field and draw the relations between each field as a graph, as seen in Fig. 6.',source=red)

# 2407.09334
p='2407.09334'; a=f'{p}:p0001:c000'
add(p,'APPLIES_TO',ent('scope','4d-n1-supergravity-prepotential','4D N = 1 supergravity prepotential formulation'),a,
    'A re-imagining of the supergravity prepotential formulation of 4D, N = 1 supergravity and its Salam-Strathdee superfield superconformal gauge group is presented.',notes='Scope properties: spacetime_dimension=4; supersymmetry=N=1.')
add(p,'USES_GROUP',ent('group','salam-strathdee-superconformal-gauge-group','Salam-Strathdee superfield superconformal gauge group'),a,
    'A re-imagining of the supergravity prepotential formulation of 4D, N = 1 supergravity and its Salam-Strathdee superfield superconformal gauge group is presented.')
add(p,'USES_GROUP',ent('group','so4','SO(4) group'),f'{p}:p0011:c000',
    'The SO(4) group tells us the same story.')
add(p,'DESCRIBES_REPRESENTATION',ent('representation','4d-lorentz-group-spectrum','4D Lorentz-group representation spectrum'),f'{p}:p0016:c000',
    'In both cases, exactly identical sets of representations of the Lorentz Groups show up at corresponding orders in the respective expansions.')
# Six explicitly enumerated supermultiplets form the paper's worked example set.
common1='Each of these adynkra genomes will be shortly shown to correspond to a previously identified 4D, N = 1 supermultiplet in the literature where the correspondence looks as: (1.) chiral supermultiplet - (3.6), (2.) 2-form gauge field supermultiplet - (3.7),'
for key,name in [('4d-n1-chiral','4D N = 1 chiral supermultiplet'),('4d-n1-two-form-gauge','4D N = 1 two-form gauge-field supermultiplet')]:
    add(p,'DESCRIBES_MULTIPLET',ent('multiplet',key,name),f'{p}:p0015:c001',common1)
common2='(3.) 1-form variant gauge field supermultiplet - (3.8), (4.) 1-form gauge field supermultiplet - (3.9), (5.) matter gravitino - (3.10), and (6.) supergravity supermultiplet - (3.11).'
for key,name in [('4d-n1-one-form-variant-gauge','4D N = 1 one-form variant gauge-field supermultiplet'),('4d-n1-one-form-gauge','4D N = 1 one-form gauge-field supermultiplet'),('4d-n1-matter-gravitino','4D N = 1 matter-gravitino supermultiplet'),('4d-n1-supergravity','4D N = 1 supergravity supermultiplet')]:
    add(p,'DESCRIBES_MULTIPLET',ent('multiplet',key,name),f'{p}:p0016:c000',common2)
overlap=ent('method','component-field-adynkra-genome-overlap','component-field and Adynkra-genome overlap')
add(p,'HAS_INPUT',ent('dataset','space-of-component-fields','space of possible component fields'),f'{p}:p0006:c000',
    'Thus, an adynkrafield results when a certain “overlap” is taken between the space of component fields and an adynkra genome.',source=overlap)
add(p,'HAS_INPUT',ent('construction','adynkra-genome','Adynkra genome'),f'{p}:p0006:c000',
    'Thus, an adynkrafield results when a certain “overlap” is taken between the space of component fields and an adynkra genome.',source=overlap)
add(p,'HAS_OUTPUT',ent('construction','adynkrafield','Adynkrafield'),f'{p}:p0006:c000',
    'Thus, an adynkrafield results when a certain “overlap” is taken between the space of component fields and an adynkra genome.',source=overlap)

for i,item in enumerate(spec,1): item['proposal_id']=f'scope-atlas-{i:03d}'
with OUT.open('w',encoding='utf-8') as f:
    for item in spec: f.write(json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
print(f'wrote {len(spec)} proposals to {OUT}')
