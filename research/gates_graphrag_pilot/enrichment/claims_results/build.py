#!/usr/bin/env python3
"""Build evidence-backed claim and result proposals for the nine-paper pilot."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNKS = Path("/tmp/gates-graphrag-pilot/chunks-enriched.jsonl")
OUTPUT = HERE / "proposals.jsonl"


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def resolve_excerpt(desired: str, chunk_text: str) -> str:
    """Return exact normalized chunk text, tolerating PDF line-end hyphenation."""
    actual = normalize(chunk_text)
    desired = normalize(desired)
    if desired in actual:
        return desired
    dehyphenated = []
    source_indexes = []
    index = 0
    while index < len(actual):
        if (
            index > 0
            and index + 2 < len(actual)
            and actual[index] == "-"
            and actual[index + 1] == " "
            and actual[index - 1].isalnum()
            and actual[index + 2].isalnum()
        ):
            index += 2
            continue
        dehyphenated.append(actual[index])
        source_indexes.append(index)
        index += 1
    collapsed = "".join(dehyphenated)
    start = collapsed.find(desired)
    if start < 0:
        raise ValueError(f"excerpt not found after dehyphenation: {desired}")
    end = start + len(desired) - 1
    return actual[source_indexes[start] : source_indexes[end] + 1]


def entity(kind: str, key: str, name: str) -> dict[str, str]:
    return {"type": kind, "key": f"{kind}:{key}", "name": name}


PAPER_TITLES = {
    "1911.00807": "Superfield Component Decompositions and the Scan for Prepotential Supermultiplets in 10D Superspaces",
    "2002.08502": "Adinkra Foundation of Component Decomposition and the Scan for Superconformal Multiplets in 11D, N = 1 Superspace",
    "2006.03609": "Advening to Adynkrafields: Young Tableaux to Component Fields of the 10D, N = 1 Scalar Superfield",
    "2007.07390": "Component Decompositions and Adynkra Libraries for Supermultiplets in Lower Dimensional Superspaces",
    "2012.13308": "The 300 “Correlators” Suggests 4D, N = 1 SUSY Is a Solution to a Set of Sudoku Puzzles",
    "2012.14015": "A Note On Exemplary Off-Shell Constructions Of 4D, N = 2 Supersymmetry Representations",
    "2304.09830": "N = 2 SUSY & the Hexipentisteriruncicantitruncated 7-Simplex",
    "2311.06842": "Unfolded Adinkra Properties of Supermultiplets (I)",
    "2407.09334": "Adynkra Genomes, Adynkrafields, and the 4D, N = 1 Supergravity Superfield Prepotential",
}


def paper(pid: str) -> dict[str, str]:
    return {"type": "paper", "key": f"arxiv:{pid}", "name": PAPER_TITLES[pid]}


# pid, chunk_id, relationship, source, target, excerpt, confidence, notes
SPECS = [
    ("1911.00807", "1911.00807:p0001:c000", "REPORTS_RESULT", paper("1911.00807"), entity("result", "complete-10d-scalar-superfield-component-descriptions", "complete Lorentz descriptions of the component fields in the N = 1, N = 2A, and N = 2B unconstrained scalar 10D superfields"), "The first complete and explicit SO(1,9) Lorentz descriptions of all component fields contained in the N = 1, N = 2A, and N = 2B unconstrained scalar 10D superfields are presented.", 0.99, None),
    ("1911.00807", "1911.00807:p0001:c000", "REPORTS_RESULT", paper("1911.00807"), entity("result", "finite-reducible-off-shell-10d-nordstrom-supergravities", "finite reducible off-shell 10D Nordström supergravity component constructions without off-shell central charges"), "A consequential deliverable of this advance is it provides the first explicit, in terms of component fields, examples of all the off-shell 10D Nordstr ̈om SG theories relevant to string theory, without off-shell central charges that are reducible but with finite numbers of fields.", 0.98, "The reported constructions are explicitly described as reducible."),
    ("1911.00807", "1911.00807:p0048:c002", "SUPPORTS", entity("result", "ten-dimensional-superfield-scan", "scan for ten-dimensional superfields containing conformal gravitons"), entity("claim", "possible-higher-dimensional-conformal-supergravity", "the ten-dimensional embeddings may lead to higher-dimensional conformal supergravity theories"), "scans suggest the possibilities of a number of superfields for embedding the component-level conformal gravitons into 10D superfields. This supports the idea of the eventual success of these efforts.", 0.94, "The paper says the scans support a possibility, not that a conformal supergravity theory was obtained."),
    ("1911.00807", "1911.00807:p0048:c001", "QUALIFIES", paper("1911.00807"), entity("claim", "possible-higher-dimensional-conformal-supergravity", "the ten-dimensional embeddings may lead to higher-dimensional conformal supergravity theories"), "Many years ago, Nahm [47] pointed out the absence of a superconformal current above six dimensions. This most certainly suggests an obstruction may exist.", 0.99, "This is an explicit obstruction noted against the proposed direction."),

    ("2002.08502", "2002.08502:p0001:c000", "REPORTS_RESULT", paper("2002.08502"), entity("result", "complete-11d-scalar-superfield-lorentz-content", "Lorentz representations of all bosonic and fermionic degrees of freedom in an unconstrained 11D scalar superfield"), "For the first time in the physics literature, the Lorentz representations of all 2,147,483,648 bosonic degrees of freedom and 2,147,483,648 fermionic degrees of freedom in an unconstrained eleven dimensional scalar superfield are presented.", 0.99, None),
    ("2002.08502", "2002.08502:p0001:c000", "REPORTS_RESULT", paper("2002.08502"), entity("result", "11d-scalar-superfield-field-and-link-counts", "field and supercharge-orbit link counts for the 11D scalar superfield"), "We find the 11D, N = 1 scalar superfield contains 1,494 bosonic fields, 1,186 fermionic fields, and a maximum number of 29,334 links connecting them via orbits of the supercharges.", 0.99, None),
    ("2002.08502", "2002.08502:p0042:c001", "MAKES_CLAIM", paper("2002.08502"), entity("claim", "11d-scalar-superfield-m-theory-limit", "conjecture that the 11D scalar superfield is an M-theory superfield limit and a prepotential or semi-prepotential"), "the facts that at the middle level of its adinkra both the conformal graviton and gauge 3-form (as well the conformal gravitino at one higher level) show up, imply V is a superfield limit of M-Theory", 0.99, "The paper explicitly labels this statement Conjecture #1 and continues by proposing V as a prepotential or semi-prepotential."),
    ("2002.08502", "2002.08502:p0042:c001", "SUPPORTS", entity("result", "11d-middle-level-supergravity-state-content", "conformal graviton, gauge three-form, and conformal gravitino content at the middle levels of the 11D scalar-superfield Adinkra"), entity("claim", "11d-scalar-superfield-m-theory-limit", "conjecture that the 11D scalar superfield is an M-theory superfield limit and a prepotential or semi-prepotential"), "the facts that at the middle level of its adinkra both the conformal graviton and gauge 3-form (as well the conformal gravitino at one higher level) show up, imply V is a superfield limit of M-Theory", 0.97, "This records the evidence expressly offered for Conjecture #1."),
    ("2002.08502", "2002.08502:p0044:c000", "QUALIFIES", paper("2002.08502"), entity("claim", "11d-scalar-superfield-m-theory-limit", "conjecture that the 11D scalar superfield is an M-theory superfield limit and a prepotential or semi-prepotential"), "the scalar superfield V contains the {1}, {65}, and {165} irreps at level-16 and contains the {32} and {320} irreps at level-17, but does not contain the {55} irrep at level-16, which suggests that V may be a semi-prepotential, i.e. some spinorial derivatives of the fundamental prepotential.", 0.99, "The added note in proof narrows the scalar-superfield interpretation toward a semi-prepotential."),

    ("2006.03609", "2006.03609:p0067:c001", "REPORTS_RESULT", paper("2006.03609"), entity("result", "adynkra-to-10d-component-field-procedure", "procedure from the 10D scalar-superfield Adynkra to component fields and their irreducibility conditions"), "In this work we have shown all the steps that allow one to begin with an adynkra of the 10D, N = 1 scalar superfield and apply a well defined set of rules to “tease” from this starting point and finally obtain the field variables (together with their irreducibility conditions) for which the Dynkin Labels provide descriptions.", 0.99, None),
    ("2006.03609", "2006.03609:p0067:c001", "QUALIFIES", paper("2006.03609"), entity("result", "adynkra-to-10d-component-field-procedure", "procedure from the 10D scalar-superfield Adynkra to component fields and their irreducibility conditions"), "However, we should remind the reader that even if this is all explicitly carried out, one still has a reducible construction. That is a separate problem needing further investigation of the properties of the quantities eG or G.", 0.99, "The component-level construction remains reducible."),
    ("2006.03609", "2006.03609:p0068:c000", "MAKES_CLAIM", paper("2006.03609"), entity("claim", "three-column-tying-rule-branching-conjecture", "conjecture that hook and tying rules calculate the stated su(N) to so(N) branching rules for Young tableaux with at most three columns"), "The calculation of the branching rules for general su(N) ⊃so(N) where AN−1 ⊃DN/2 for even N, or AN−1 ⊃B(N−1)/2 for odd N, may be found by using the hook rule and the application of the tying rules for that irrep’s Young Tableau in su(N), if that Young Tableau contains less than or equal to three columns.", 0.99, "The paper expressly casts this statement as a conjecture."),
    ("2006.03609", "2006.03609:p0068:c000", "QUALIFIES", paper("2006.03609"), entity("claim", "three-column-tying-rule-branching-conjecture", "conjecture that hook and tying rules calculate the stated su(N) to so(N) branching rules for Young tableaux with at most three columns"), "However, this still is not a replacement for a rigorous mathematical proof.", 1.0, "The paper explicitly states the evidentiary limitation."),

    ("2007.07390", "2007.07390:p0055:c001", "REPORTS_RESULT", paper("2007.07390"), entity("result", "lower-dimensional-adynkra-libraries", "Adynkra libraries for Lorentzian spacetimes in dimensions four through nine"), "In this work, we have established the basic libraries of adynkras that can be used to explore problems of embedding component fields into superfields in the context of spacetimes with Lorentzian signature and D −1 spatial dimensions where 4 ≤D ≤9.", 0.99, None),
    ("2007.07390", "2007.07390:p0055:c001", "QUALIFIES", paper("2007.07390"), entity("result", "lower-dimensional-adynkra-libraries", "Adynkra libraries for Lorentzian spacetimes in dimensions four through nine"), "A fully satisfactory answer would require the construction of supercovariant derivative operators acting on adynkrafields.", 0.99, "This qualifies the libraries with respect to complete supersymmetry transformations."),
    ("2007.07390", "2007.07390:p0055:c001", "MAKES_CLAIM", paper("2007.07390"), entity("claim", "algorithmic-superfield-embedding-from-component-spectrum", "an algorithm can derive minimal-dimension superfield representations containing a specified component-field spectrum"), "One can now start with any spectrum of on-shell component fields in these various dimensions and algorithmically derive the minimal dimension superfield representation (as well as alternatives) that contains this specified component field spectrum.", 0.97, "This is the capability asserted for the Adynkra digital analysis scan."),

    ("2012.13308", "2012.13308:p0001:c000", "MAKES_CLAIM", paper("2012.13308"), entity("claim", "susy-weight-space-embedded-in-permutahedra", "conjecture that the weight space of 4D N-extended supersymmetry representations is embedded in permutahedra associated with permutation groups"), "A conjecture is made that the weight space for 4D, N-extended supersymmetrical representations is embedded within the permutahedra associated with permutation groups Sd.", 1.0, "The abstract explicitly calls this a conjecture."),
    ("2012.13308", "2012.13308:p0001:c000", "REPORTS_RESULT", paper("2012.13308"), entity("result", "four-color-truncated-octahedron-equivalence", "equivalence between minimal off-shell 4D N = 1 supersymmetry mathematics and a four-color problem on the truncated octahedron"), "It is shown that the appearance of the mathematics of 4D, N = 1 minimal off-shell supersymmetry representations is equivalent to solving a four color problem on the truncated octahedron.", 0.99, None),
    ("2012.13308", "2012.13308:p0023:c001", "DERIVES", paper("2012.13308"), entity("result", "thirty-inter-quartet-correlator-matrices", "thirty inter-quartet correlator matrices from six ordered source and target quartets"), "From simply counting 6 × 5 = 30, we can create in total 30 inter-quartet correlator matrices.", 0.99, None),
    ("2012.13308", "2012.13308:p0032:c002", "REPORTS_RESULT", paper("2012.13308"), entity("result", "susy-quartets-maximize-correlator-eigenvector-length", "SUSY quartets in S4 attain the maximum quartet-correlator eigenvector length"), "Thus, we conclude that the SUSY quartets in S4 are precisely the ones that lead to the maximum possible value of the lengths of the eigenvectors for quartet correlators.", 0.99, None),
    ("2012.13308", "2012.13308:p0033:c000", "MAKES_CLAIM", paper("2012.13308"), entity("claim", "susy-representation-rules-as-sudoku", "proposal that SUSY representation theory can be studied as diadem-embedding Sudoku problems in permutation groups"), "We believe the most powerful implication of the observations in this work is that the representation theory for SUSY for all values of N can be interpreted a Sudoku puzzle where the diadems set the start of rules.", 0.97, "The authors frame this as their interpretation of the observations."),

    ("2012.14015", "2012.14015:p0001:c000", "REPORTS_RESULT", paper("2012.14015"), entity("result", "hymn-selects-off-shell-4d-n2-combinations", "HYMN classification selects combinations of off-shell 4D N = 1 supermultiplets corresponding to off-shell 4D N = 2 supermultiplets"), "Even though the HYMN definition was designed to distinguish between the raising and lowering of nodes in one dimensional valise supermultiplets, they are shown to accurately select out which combinations of off-shell 4D, N = 1 supermultiplets correspond to off-shell 4D, N = 2 supermultiplets.", 0.99, None),
    ("2012.14015", "2012.14015:p0001:c000", "REPORTS_RESULT", paper("2012.14015"), entity("result", "chiral-vector-and-chiral-tensor-share-valise-class", "only the chiral-vector and chiral-tensor combinations have valises in the same class"), "Only the combinations of the chiral + vector and chiral + tensor are found to have valises in the same class.", 1.0, None),

    ("2304.09830", "2304.09830:p0027:c000", "REPORTS_RESULT", paper("2304.09830"), entity("result", "four-color-to-eight-color-constructive-rules", "constructive rules from four-color Adinkras and Clifford algebras to eight-color systems"), "In Sec. 2, the focus was on how previously uncovered adinkras and associated Clifford Algebras in the case of four colors, can be as a starting to construct the similar adinkras and associated Clifford Algebras in the case of eight colors. We demonstrated a set of constructive rules that accomplish this goal.", 0.98, None),
    ("2304.09830", "2304.09830:p0028:c000", "REPORTS_RESULT", paper("2304.09830"), entity("result", "hopping-operators-generate-permutahedron-translations", "left and right hopping operators that generate translations on the permutahedron"), "Using these concepts, we constructed elementary left hopping and right hopping operators which have the effect of generating translation on the space of the permutahedron.", 0.99, None),
    ("2304.09830", "2304.09830:p0028:c000", "REPORTS_RESULT", paper("2304.09830"), entity("result", "magic-number-values-s4-s8-s16", "reported magic-number values 12, 112, and 960 for S4, S8, and S16"), "For the cases of {S4}, {S8}, and {S16}, our studies indicate the results shown in Table. 5 G(3)-Value Permutation Group 12 {S4} 112 {S8} 960 {S16}", 0.99, None),
    ("2304.09830", "2304.09830:p0028:c000", "MAKES_CLAIM", paper("2304.09830"), entity("claim", "magic-number-formula-for-s2r", "conjectured magic-number formula for the special permutation groups S(2r)"), "and we conjecture the formula in equation (3.28) will hold for the special cases of {S2r}.", 1.0, "The statement is explicitly conjectural."),
    ("2304.09830", "2304.09830:p0028:c001", "REPORTS_RESULT", paper("2304.09830"), entity("result", "one-hundred-sixty-eight-ab-normal-subsets-s8", "168 S8 subsets that act as normal groups when treated as sets"), "But remarkably enough when one looks at sets instead of individual elements, there are one-hundred and sixty-eight subsets that act as if they are normal groups. We have given the name ‘ab-normal’ to these sets.", 0.99, None),
    ("2304.09830", "2304.09830:p0028:c001", "REPORTS_RESULT", paper("2304.09830"), entity("result", "partial-k-face-spectrum-through-n8-k5", "computer-derived partial k-face spectrum through N = 8 and k = 5"), "By using the computer simulation, we obtain the partial k-face spectrum up to N = 8 and k = 5.", 0.99, "The result is explicitly partial."),
    ("2304.09830", "2304.09830:p0029:c000", "MAKES_CLAIM", paper("2304.09830"), entity("claim", "thirty-decomposable-n8-pairings", "thirty pairings form the basis for decomposable 1D N = 8 off-shell supermultiplets"), "So there must be 6 × 5 = 30 such pairings as the basis to construct 1D, N = 8 off-shell supermultiplets that are decomposable!", 0.98, "This count is limited to the decomposable construction described in the paper."),
    ("2304.09830", "2304.09830:p0029:c000", "REQUIRES_ASSUMPTION", entity("claim", "thirty-decomposable-n8-pairings", "thirty pairings form the basis for decomposable 1D N = 8 off-shell supermultiplets"), entity("scope", "distinct-four-color-subset-pairs", "each member of a pair of four-color subsets is distinct"), "each member of the pairs must be chosen to be distinct. So there must be 6 × 5 = 30 such pairings as the basis to construct 1D, N = 8 off-shell supermultiplets that are decomposable!", 0.99, "This records the premise used for the count of ordered pairings."),

    ("2311.06842", "2311.06842:p0042:c001", "REPORTS_RESULT", paper("2311.06842"), entity("result", "unfolded-adinkras-for-cs-vs-ts-cls", "unfolded Adinkras for the chiral, vector, tensor, and complex-linear supermultiplet networks"), "In this paper, we have defined and constructed unfolded adinkras for the CS, VS, TS, and CLS networks.", 1.0, None),
    ("2311.06842", "2311.06842:p0042:c001", "REPORTS_RESULT", paper("2311.06842"), entity("result", "unfolded-adinkra-periodicity-and-modified-garden-algebra", "periodicity after the fifth unfolded-Adinkra level and satisfaction of the modified Garden algebra"), "One can show there is a periodicity on the adinkra connection after the fifth level. Compared to the folded adinkra construction process, we can also verify that the modified garden algebra holds for the unfolded field definitions.", 0.98, None),
    ("2311.06842", "2311.06842:p0039:c001", "REPORTS_RESULT", paper("2311.06842"), entity("result", "echi-one-equals-echi-two", "eχ(1) equals eχ(2) for the analyzed systems"), "The set of calculations can be carried out at the second level of the disaggregated adinkras to define eχ(2). It is easily seen that the same results are obtained. So in these systems, we have eχ(1) = eχ(2)", 0.99, None),
    ("2311.06842", "2311.06842:p0039:c002", "QUALIFIES", paper("2311.06842"), entity("result", "echi-one-equals-echi-two", "eχ(1) equals eχ(2) for the analyzed systems"), "It should be noted that the values of eχ(1) and eχ(2) values are specific to our field definitions.", 1.0, "The reported values depend on the selected field definitions."),
    ("2311.06842", "2311.06842:p0039:c002", "MAKES_CLAIM", paper("2311.06842"), entity("claim", "echi-level-equality-for-valise-definitions", "conjecture that eχ(1) equals eχ(2) when supermultiplet field definitions are in valise form"), "However, we conjecture that Eq. (6.4) will hold if the supermultiplet field definitions are in valise form.", 1.0, "The statement is explicitly conjectural and conditional on valise field definitions."),
    ("2311.06842", "2311.06842:p0040:c000", "MAKES_CLAIM", paper("2311.06842"), entity("claim", "chi-values-encode-4d-lorentz-representation", "the graph quantities χ and eχ encode the Lorentz representation of the analyzed 4D N = 1 systems"), "Thus, the Lorentz representation of the 4D, N = 1 system is distinctly encoded in the values different values of the graph-theoretic quantities χ and eχ.", 0.98, None),
    ("2311.06842", "2311.06842:p0042:c000", "MAKES_CLAIM", paper("2311.06842"), entity("claim", "zero-net-nodal-vorticity-for-all-adinkras", "conjecture that every Adinkra has zero net nodal vorticity at every level"), "Based on this observation and the examination of some other examples, we conjecture that this result is valid for all adinkras.", 0.98, "The universal statement is explicitly a conjecture based on examples."),

    ("2407.09334", "2407.09334:p0019:c001", "REPORTS_RESULT", paper("2407.09334"), entity("result", "adynkrafield-consistent-with-4d-supergravity-prepotential", "consistency of the Adynkrafield concept with the 4D N = 1 Salam-Strathdee supergravity prepotential and superconformal group descriptions"), "In this work, we have undertaken the task of showing that the concept of an “adynkrafield” is consistent with the introduction of the 4D, N = 1 Salam-Strathdee superspace definition of the supergravity prepotential as well as its associate superfield description of the superconformal group.", 0.99, None),
    ("2407.09334", "2407.09334:p0019:c001", "MAKES_CLAIM", paper("2407.09334"), entity("claim", "duplicate-level-representations-indicate-reducibility", "conjecture that duplicate representations at a level are necessary but not sufficient evidence of reducibility in an Adynkra representation"), "We conjecture that such occurrences are a necessary, but not sufficient, feature indicating reducibility of any adynkra representation.", 1.0, "The statement is explicitly conjectural and states both necessity and insufficiency."),
    ("2407.09334", "2407.09334:p0019:c001", "QUALIFIES", paper("2407.09334"), entity("claim", "duplicate-level-representations-indicate-reducibility", "conjecture that duplicate representations at a level are necessary but not sufficient evidence of reducibility in an Adynkra representation"), "Thus, the next great challenge will be to marshal ideas about how to accomplish the complete reduction of U into an irreducible representation of SUSY.", 0.97, "The paper states that the actual irreducible reduction remains unresolved."),
]


def main() -> int:
    chunks = {}
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            chunks[row["chunk_id"]] = row

    proposals = []
    for index, (pid, chunk_id, relationship, source, target, excerpt, confidence, notes) in enumerate(SPECS, 1):
        chunk = chunks[chunk_id]
        try:
            excerpt = resolve_excerpt(excerpt, chunk["text"])
        except ValueError as error:
            raise ValueError(f"claims-results-{index:03d} in {chunk_id}: {error}") from error
        proposal = {
            "proposal_id": f"claims-results-{index:03d}",
            "source": source,
            "relationship": relationship,
            "target": target,
            "evidence": {
                "paper_id": pid,
                "chunk_id": chunk_id,
                "page_number": chunk["page_number"],
                "section": chunk.get("section_heading"),
                "excerpt": excerpt,
            },
            "basis": "explicit_text",
            "review_status": "pending",
            "confidence": confidence,
        }
        if notes:
            proposal["notes"] = notes
        proposals.append(proposal)

    HERE.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(proposals)} proposals to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
