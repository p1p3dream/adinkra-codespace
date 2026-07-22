#!/usr/bin/env python3
"""Build evidence-backed concept and method proposals for the nine-paper pilot."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
CHUNKS = Path("/tmp/gates-graphrag-pilot/chunks-enriched.jsonl")
MANIFEST = ROOT / "research/gates_graphrag_pilot/metadata/manifest.csv"
OUT = Path(__file__).with_name("proposals.jsonl")
ALIASES_OUT = Path(__file__).with_name("ENTITY_ALIASES.json")


EXTRA_ALIASES = {
    "concept:ten-dimensional-adinkra-graph": ["10D Adinkra graph", "ten dimensional Adinkra graph"],
    "method:lie-algebra-branching-rule": ["branching rule", "branching rules"],
    "method:breitenlohner-prepotential-scan": ["Breitenlohner approach", "Breitenlohner scan"],
    "concept:fermionic-young-tableau": ["fermionic Young tableaux", "FYT"],
    "method:higher-dimensional-adinkra-technology": ["higher dimensional Adinkra technology"],
    "concept:eleven-dimensional-unconstrained-scalar-superfield": ["11D unconstrained scalar superfield", "11D, N = 1 scalar superfield"],
    "method:adinkra-based-component-decomposition": ["Adinkra based component decomposition"],
    "algorithm:projection-matrix-branching-algorithm": ["projection matrix branching algorithm"],
    "method:plethysm": ["plethysms"],
    "concept:adynkra": ["adynkras"],
    "method:tying-rule": ["tying rules"],
    "concept:adynkrafield": ["adynkrafields"],
    "method:young-tableau-expansion": ["Young Tableaux expansion", "Young Tableaux expansions"],
    "concept:level-parameter": ["level parameters"],
    "concept:mixed-young-tableau": ["mixed Young tableaux", "mixed YT"],
    "concept:adynkra-library": ["Adynkra Libraries", "Adynkra libraries"],
    "concept:component-field-multiplet-embedding": ["component field multiplet embedding"],
    "method:dynkin-young-field-translation": ["Dynkin label to Young tableau to field variable translation", "DL-YT-field translation"],
    "method:adynkra-diagram-encoding": ["Adynkra diagram encoding"],
    "concept:supersymmetry-weight-space-in-permutahedron": ["SUSY weight-space embedding in a permutahedron"],
    "method:four-color-truncated-octahedron-problem": ["four color problem on the truncated octahedron"],
    "method:weak-bruhat-ordering": ["weak Bruhat order"],
    "concept:s4-permutahedron-dissection": ["S4 permutahedron dissection"],
    "concept:s8-permutation": ["S8 permutations"],
    "invariant:height-yielding-matrix-number": ["Height Yielding Matrix Numbers", "HYMN", "HYMNs", "HYMN value", "HYMN values"],
    "algorithm:recursive-supermultiplet-construction": ["recursive construction of supermultiplets"],
    "concept:hopping-operator": ["hopping operators", "hopper", "hoppers", "left hopping operator", "right hopping operator"],
    "method:truncation-and-chromatic-number": ["truncation and chromatic number"],
    "concept:ab-normal-coset": ["ab-normal cosets"],
    "method:computer-simulation-of-permutahedron-faces": ["permutahedron face simulation"],
    "concept:infinite-unfolded-adinkra": ["infinite unfolded Adinkras", "infinitely unfolded Adinkra", "infinitely unfolded Adinkras"],
    "concept:unfolded-adinkra": ["unfolded Adinkras"],
    "invariant:net-centric-chi-1": ["eχ(1)", "net-centric eχ(1)"],
    "invariant:net-centric-chi-2": ["eχ(2)", "net-centric eχ(2)"],
    "invariant:adinkra-vorticity": ["Adinkra vorticity", "nodal vorticity"],
    "method:time-derivative-unfolding": ["time derivative unfolding"],
    "concept:four-dimensional-supergravity-prepotential": ["4D, N = 1 supergravity superfield prepotential"],
    "method:bosonic-and-spinorial-young-tableaux": ["bosonic and spinorial YT", "blue and red Young tableaux"],
    "method:wedge-product": ["wedge products"],
    "concept:adynkra-genome": ["Adynkra Genome", "Adynkra Genomes", "AG"],
    "assumption:nilpotent-level-parameter": ["level-parameter nilpotency", "nilpotency of level parameters"],
    "algorithm:adynkra-genome-construction": ["Adynkra Genome construction", "AG construction"],
    "method:adynkrafield-overlap-construction": ["adynkrafield overlap"],
}


def norm(text: str) -> str:
    return " ".join(text.split())


# (chunk_id, relationship, target type, target key, target name,
#  excerpt start, excerpt end, confidence, optional note)
SPECS = [
    ("1911.00807:p0001:c000", "DEFINES", "concept", "ten-dimensional-adinkra-graph", "ten-dimensional adinkra graph", "Adinkra graphs for ten dimensional", "SO(1,9).", 0.99, None),
    ("1911.00807:p0001:c000", "USES", "method", "lie-algebra-branching-rule", "Lie-algebra branching rule", "These are made possible", "subalgebras.", 0.97, None),
    ("1911.00807:p0001:c000", "USES", "method", "breitenlohner-prepotential-scan", "Breitenlohner prepotential scan", "An analogue of Breitenlohner", "Yang-Mills theories.", 0.98, None),
    ("1911.00807:p0004:c001", "DEFINES", "concept", "fermionic-young-tableau", "fermionic Young tableau", "We will call the former", "(BYT).", 0.99, "The passage distinguishes fermionic Young tableaux from bosonic Young tableaux."),
    ("1911.00807:p0005:c000", "INTRODUCES", "method", "higher-dimensional-adinkra-technology", "higher-dimensional adinkra technology", "Chapter three is a transitional", "higher dimensional adinkra technology.", 0.95, None),

    ("2002.08502:p0001:c000", "STUDIES", "concept", "eleven-dimensional-unconstrained-scalar-superfield", "eleven-dimensional unconstrained scalar superfield", "For the ﬁrst time", "are presented.", 0.99, None),
    ("2002.08502:p0001:c000", "STUDIES", "method", "adinkra-based-component-decomposition", "adinkra-based component decomposition", "Comparisons of the conceptual", "over the oth- ers.", 0.96, "The PDF text contains a line-break hyphen in 'others'."),
    ("2002.08502:p0026:c001", "DEFINES", "concept", "lie-algebra-branching-rule", "Lie-algebra branching rule", "First, a branching rule", "subalgebras h.", 0.99, None),
    ("2002.08502:p0026:c001", "DEFINES", "algorithm", "projection-matrix-branching-algorithm", "projection-matrix branching algorithm", "Thus, the algorithm for calculating", "projected weight diagram.", 0.98, None),
    ("2002.08502:p0029:c001", "DEFINES", "method", "plethysm", "plethysm", "Generally, branching rules", "Plethysm [32,33,34].", 0.99, None),
    ("2002.08502:p0029:c001", "USES", "method", "plethysm", "plethysm", "Therefore, the component decomposition", "(00001).", 0.96, None),

    ("2006.03609:p0007:c002", "DEFINES", "concept", "adynkra", "adynkra", "Henceforth, we will refer", "from “Dynkin.”", 0.99, None),
    ("2006.03609:p0001:c000", "INTRODUCES", "method", "tying-rule", "tying rule", "In order to reach", "Schur function series.", 0.99, None),
    ("2006.03609:p0001:c000", "INTRODUCES", "concept", "adynkrafield", "adynkrafield", "The expansions are given", "Dynkin Labels.", 0.99, None),
    ("2006.03609:p0001:c000", "USES", "method", "young-tableau-expansion", "Young-tableau expansion", "We show this is possible", "by Dynkin Labels.", 0.98, None),
    ("2006.03609:p0005:c000", "INTRODUCES", "concept", "level-parameter", "level parameter", "The importance of two", "basis for expan- sions.", 0.95, "The PDF text contains a line-break hyphen in 'expansions'."),
    ("2006.03609:p0006:c000", "INTRODUCES", "concept", "mixed-young-tableau", "mixed Young tableau", "Mixed Young Tableaux as", "are introduced.", 0.98, None),

    ("2007.07390:p0001:c000", "INTRODUCES", "concept", "adynkra-library", "Adynkra library", "We present Adynkra Libraries", "dimension nine through four.", 0.99, None),
    ("2007.07390:p0001:c000", "STUDIES", "concept", "component-field-multiplet-embedding", "component-field multiplet embedding", "that can be used", "dimension nine through four.", 0.98, None),
    ("2007.07390:p0006:c002", "USES", "method", "lie-algebra-branching-rule", "Lie-algebra branching rule", "su(d) ⊃so(1, D −1)", "LieART [11].", 0.97, None),
    ("2007.07390:p0006:c002", "USES", "method", "plethysm", "plethysm", "The other one involves", "by Susyno [10].", 0.99, None),
    ("2007.07390:p0006:c002", "USES", "method", "dynkin-young-field-translation", "Dynkin-label to Young-tableau to field-variable translation", "so that we can translate", "irreducible conditions.", 0.98, None),
    ("2007.07390:p0006:c002", "USES", "method", "adynkra-diagram-encoding", "Adynkra-diagram encoding", "One can draw adynkra", "Section 8.3 of [3].", 0.98, None),

    ("2012.13308:p0001:c000", "STUDIES", "concept", "supersymmetry-weight-space-in-permutahedron", "supersymmetry weight-space embedding in a permutahedron", "A conjecture is made", "permutation groups Sd.", 0.96, "The edge records the subject studied, not acceptance of the conjecture."),
    ("2012.13308:p0001:c000", "STUDIES", "method", "four-color-truncated-octahedron-problem", "four-color truncated-octahedron problem", "It is shown that", "truncated octahedron.", 0.99, None),
    ("2012.13308:p0003:c001", "USES", "method", "weak-bruhat-ordering", "weak Bruhat ordering", "The dissection can be", "weak Bruhat ordering.", 0.99, None),
    ("2012.13308:p0003:c001", "STUDIES", "concept", "s4-permutahedron-dissection", "S4 permutahedron dissection", "One of the main", "properties of S4.", 0.98, None),

    ("2012.14015:p0001:c000", "STUDIES", "concept", "s8-permutation", "S8 permutation", "We study the S8", "N = 2 supermultiplets.", 0.99, None),
    ("2012.14015:p0001:c000", "USES", "invariant", "height-yielding-matrix-number", "Height Yielding Matrix Number", "Even though the HYMN", "N = 2 super- multiplets.", 0.98, "The PDF text contains a line-break hyphen in 'supermultiplets'."),
    ("2012.14015:p0005:c000", "DEFINES", "invariant", "height-yielding-matrix-number", "Height Yielding Matrix Number", "This leads the way", "each supermultiplet.", 0.99, None),

    ("2304.09830:p0001:c000", "STUDIES", "algorithm", "recursive-supermultiplet-construction", "recursive supermultiplet construction", "We study algorithms for", "N = 1 supermul- tiplet matrices.", 0.99, "The PDF text contains a line-break hyphen in 'supermultiplet'."),
    ("2304.09830:p0001:c000", "INTRODUCES", "concept", "hopping-operator", "hopping operator", "The concept of ‘hopping", "the permutahedron.", 0.99, None),
    ("2304.09830:p0016:c000", "DEFINES", "concept", "hopping-operator", "hopping operator", "Now we are in", "right hopping operators.", 0.99, None),
    ("2304.09830:p0001:c000", "USES", "method", "truncation-and-chromatic-number", "truncation and chromatic-number analysis", "We observe connections between", "chromatic number.", 0.99, None),
    ("2304.09830:p0001:c000", "INTRODUCES", "concept", "ab-normal-coset", "ab-normal coset", "Although these hopping operators", "unordered sets.", 0.98, None),
    ("2304.09830:p0001:c000", "USES", "method", "computer-simulation-of-permutahedron-faces", "computer simulation of permutahedron faces", "Finally, using computer simulations", "lower-order supermultiplets.", 0.99, None),

    ("2311.06842:p0001:c000", "INTRODUCES", "concept", "infinite-unfolded-adinkra", "infinite unfolded Adinkra", "We call these “infinite", "N = 1 supermultiplets.", 0.99, None),
    ("2311.06842:p0004:c001", "DEFINES", "concept", "unfolded-adinkra", "unfolded Adinkra", "There is a hitherto", "engineering dimensions [12].", 0.99, None),
    ("2311.06842:p0001:c000", "INTRODUCES", "invariant", "net-centric-chi-1", "net-centric chi-1 quantity", "New “net-centric” quantities", "N = 1 theories.", 0.99, "The cited sentence introduces both eχ(1) and eχ(2)."),
    ("2311.06842:p0001:c000", "INTRODUCES", "invariant", "net-centric-chi-2", "net-centric chi-2 quantity", "New “net-centric” quantities", "N = 1 theories.", 0.99, "The cited sentence introduces both eχ(1) and eχ(2)."),
    ("2311.06842:p0039:c001", "DEFINES", "invariant", "net-centric-chi-1", "net-centric chi-1 quantity", "We define a numerical", "first level.", 0.99, None),
    ("2311.06842:p0039:c001", "DEFINES", "invariant", "net-centric-chi-2", "net-centric chi-2 quantity", "The set of calculations", "define eχ(2).", 0.99, None),
    ("2311.06842:p0001:c000", "INTRODUCES", "invariant", "adinkra-vorticity", "Adinkra vorticity", "A pre- viously unobserved", "is noted.", 0.98, "The PDF text contains a line-break hyphen in 'previously'."),
    ("2311.06842:p0041:c001", "DEFINES", "invariant", "adinkra-vorticity", "Adinkra vorticity", "We define vorticity", "fermionic nodes.", 0.99, None),
    ("2311.06842:p0045:c000", "USES", "method", "time-derivative-unfolding", "time-derivative unfolding", "This calculation shows", "N = 1 Chiral supermultiplet.", 0.99, None),

    ("2407.09334:p0001:c000", "STUDIES", "concept", "four-dimensional-supergravity-prepotential", "4D, N = 1 supergravity prepotential", "A re-imagining of", "is presented.", 0.99, None),
    ("2407.09334:p0004:c001", "USES", "method", "bosonic-and-spinorial-young-tableaux", "bosonic and spinorial Young tableaux", "The 4D, N = 1 adynkra", "“red boxes” .", 0.98, None),
    ("2407.09334:p0005:c000", "USES", "method", "wedge-product", "wedge product", "In order to simulate", "“adynkra genome”.", 0.99, None),
    ("2407.09334:p0005:c000", "USES", "method", "plethysm", "plethysm", "This is accomplished by", "(see chapter 3).", 0.99, None),
    ("2407.09334:p0005:c000", "DEFINES", "concept", "adynkra-genome", "Adynkra genome", "A genome here may", "corresponding adynkra.", 0.99, None),
    ("2407.09334:p0005:c000", "ASSUMES", "assumption", "nilpotent-level-parameter", "nilpotent level parameter", "We postulate that nilpotency", "among their properties.", 0.99, "The paper labels this condition as a postulate."),
    ("2407.09334:p0014:c000", "DEFINES", "algorithm", "adynkra-genome-construction", "Adynkra-genome construction", "The first operator required", "space of YT.", 0.99, None),
    ("2407.09334:p0016:c001", "DEFINES", "method", "adynkrafield-overlap-construction", "adynkrafield overlap construction", "Passing from an adynkra", "denoted by {F}.", 0.99, None),
]


def excerpt_between(text: str, start: str, end: str) -> str:
    start_at = text.find(start)
    if start_at < 0:
        raise ValueError(f"start marker not found: {start!r}")
    end_at = text.find(end, start_at)
    if end_at < 0:
        raise ValueError(f"end marker not found: {end!r}")
    return text[start_at : end_at + len(end)]


def main() -> None:
    chunks = {
        row["chunk_id"]: row
        for row in (json.loads(line) for line in CHUNKS.read_text().splitlines())
    }
    with MANIFEST.open(newline="") as handle:
        titles = {row["arxiv_id"]: row["title"] for row in csv.DictReader(handle)}

    proposals = []
    for index, spec in enumerate(SPECS, 1):
        chunk_id, relationship, target_type, target_key, target_name, start, end, confidence, notes = spec
        chunk = chunks[chunk_id]
        paper_id = chunk["paper_id"]
        text = norm(chunk["text"])
        excerpt = excerpt_between(text, start, end)
        proposal = {
            "proposal_id": f"concepts-methods-{index:03d}",
            "source": {
                "type": "paper",
                "key": f"arxiv:{paper_id}",
                "name": titles[paper_id],
            },
            "relationship": relationship,
            "target": {
                "type": target_type,
                "key": f"{target_type}:{target_key}",
                "name": target_name,
            },
            "evidence": {
                "paper_id": paper_id,
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

    OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in proposals)
    )
    targets = {}
    for row in proposals:
        target = row["target"]
        targets[target["key"]] = {
            "key": target["key"],
            "type": target["type"],
            "name": target["name"],
            "aliases": sorted(set(EXTRA_ALIASES.get(target["key"], []))),
        }
    alias_document = {
        "schema_version": "gates-entity-aliases-v1",
        "normalization_policy": "Conservative aliases only; no cross-concept equivalence is inferred.",
        "entities": [targets[key] for key in sorted(targets)],
    }
    ALIASES_OUT.write_text(
        json.dumps(alias_document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(proposals)} proposals to {OUT}")
    print(f"wrote {len(targets)} canonical entities to {ALIASES_OUT}")


if __name__ == "__main__":
    main()
