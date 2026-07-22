/**
 * the_168_narrative.mjs
 *
 * The nine-beat guided demo, "From 24 to 168". Each beat is a pure
 * data record: which dataset to show, the visual-state targets to
 * tween toward, which vertices to light, and one short caption whose
 * claims stay inside what the checked-in JSON actually certifies.
 * The shell (the_168.html) walks these records; nothing here touches
 * the DOM or WebGL directly.
 */

const S8_TITLE_TARGETS = {
  fog: 1, right: 0, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0,
  pointSize: 2.3, pointAlpha: 0.34, edgeAlpha: 0.028, scale: 1.3, rotorSpeed: 1,
};

/**
 * Build the beat list from the loaded data. s8 and s4 are atlas
 * objects, garden is the Garden scan record.
 */
export function makeBeats({ s8, s4, garden }) {
  const reps = new Map(s8.representations.map(rep => [rep.id, rep]));
  const octets = new Map(garden.named_octets.map(o => [o.id, o]));

  const closing = garden.named_octets.find(o => o.id === "CT" && o.published_status === "published_garden_closure")
    || garden.named_octets.find(o => o.published_status === "published_garden_closure");
  const nonclosing = garden.named_octets.find(o => o.id === "CC" && o.published_status === "published_nonclosure")
    || garden.named_octets.find(o => o.published_status === "published_nonclosure");

  const highlightOctet = (id, code) => {
    const rep = reps.get(id);
    if (!rep) return null;
    return new Map(rep.member_ranks.map(rank => [rank, code]));
  };

  const nrm = garden.normalizer;

  return [
    {
      id: 0,
      title: "The 168",
      dataset: "s8",
      caption: `${s8.metadata.vertex_count.toLocaleString()} vertices, `
        + `${s8.metadata.edge_count.toLocaleString()} edges. Every vertex is a `
        + `permutation of (1..8), a point in R^8 on the hyperplane sum = 36. `
        + `The object is genuinely 7 dimensional; it is drifting through SO(7) right now. `
        + `Drag to steer the rotation. Somewhere inside it, 168 octets are waiting.`,
      targets: { ...S8_TITLE_TARGETS },
      highlights: null,
      verdict: null,
    },
    {
      id: 1,
      title: "First, one you can hold",
      dataset: "s4",
      caption: `S4: ${s4.metadata.vertex_count} permutations of (1..4), `
        + `${s4.metadata.edge_count} edges. This permutahedron is genuinely 3 dimensional, `
        + `the truncated octahedron. Every idea in this instrument, cosets, distances, `
        + `signings, is already visible here at a size the eye can audit.`,
      targets: {
        fog: 1, right: 0, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0,
        pointSize: 9, pointAlpha: 0.95, edgeAlpha: 0.4, scale: 2.6, rotorSpeed: 0.45,
      },
      highlights: null,
      verdict: null,
    },
    {
      id: 2,
      title: "Measure something",
      dataset: "s4",
      caption: `Click any two vertices. The number that appears is their minimal `
        + `weak-Bruhat distance: the fewest adjacent swaps from one to the other. `
        + `Honesty note, quoting the dataset itself: this is `
        + `"${s8.metadata.correlator_definition}". A distance table, not physics.`,
      targets: {
        fog: 1, right: 0, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0,
        pointSize: 9, pointAlpha: 0.95, edgeAlpha: 0.4, scale: 2.6, rotorSpeed: 0.25,
      },
      highlights: null,
      verdict: null,
      measure: true,
    },
    {
      id: 3,
      title: "The ascent: 24 to 40,320",
      dataset: "s8",
      caption: `Same construction, n = 8. The 24 becomes 40,320, and the 3 dimensional `
        + `solid becomes a 7 dimensional one. What you see is an exact 3D orthographic `
        + `slice of that object; the rotation you are watching happens in all 7 dimensions.`,
      enterSnap: { scale: 0.12, pointAlpha: 0.06, edgeAlpha: 0.0 },
      targets: { ...S8_TITLE_TARGETS, rotorSpeed: 0.8 },
      highlights: null,
      verdict: null,
    },
    {
      id: 4,
      title: "Crystallize: 5,040 octets",
      dataset: "s8",
      caption: `Right multiplication by the eight elements of R8 slices all 40,320 `
        + `vertices into ${s8.right_slices.length.toLocaleString()} right cosets of 8, `
        + `the octets. Watch the fog pull toward each octet's centroid: the structure `
        + `was there all along, the coloring just lets you see it.`,
      targets: {
        fog: 0, right: 1, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0.55,
        pointSize: 3.1, pointAlpha: 0.8, edgeAlpha: 0.02, scale: 1.3, rotorSpeed: 0.55,
      },
      highlights: null,
      verdict: null,
    },
    {
      id: 5,
      title: "Left against right",
      dataset: "s8",
      caption: `R8 also slices S8 from the left. Here the right coloring (cyan to violet) `
        + `interferes with the left coloring (green to teal). For most vertices the two `
        + `slicings disagree. The brightened vertices sit where a right coset IS a left `
        + `coset, and there are exactly 168 such octets. Hold that number.`,
      targets: {
        fog: 0, right: 1, left: 0, moire: 1, foreshadow: 0.85, gold: 0, gather: 0.55,
        pointSize: 3.1, pointAlpha: 0.85, edgeAlpha: 0.02, scale: 1.3, rotorSpeed: 0.4,
      },
      highlights: null,
      verdict: null,
    },
    {
      id: 6,
      title: "A named octet",
      dataset: "s8",
      caption: (() => {
        const rep = reps.get("CC");
        return `The published literature names specific octets. Lit in white: `
          + `${rep.label}, eight vertices, one right coset `
          + `(${rep.source}). Its addresses run ${rep.member_addresses[0]} to `
          + `${rep.member_addresses[7]}. Seven such named octets are in this atlas.`;
      })(),
      targets: {
        fog: 0.3, right: 0.4, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0.55,
        pointSize: 2.8, pointAlpha: 0.35, edgeAlpha: 0.015, scale: 1.3, rotorSpeed: 0.3,
      },
      highlights: () => highlightOctet("CC", 1),
      verdict: null,
    },
    {
      id: 7,
      title: "Closure is particular",
      dataset: "s8",
      caption: `Green: ${closing.id}, published status "${closing.published_status}". `
        + `Amber: ${nonclosing.id}, published status "${nonclosing.published_status}". `
        + `Every one of the 5,040 octets admits SOME Garden signing `
        + `(rank 45, nullity 19, 2^19 solutions each). But, quoting the scan: `
        + `"${garden.boundary}" The published nonclosure stands.`,
      targets: {
        fog: 0.3, right: 0.3, left: 0, moire: 0, foreshadow: 0, gold: 0, gather: 0.55,
        pointSize: 2.8, pointAlpha: 0.3, edgeAlpha: 0.015, scale: 1.3, rotorSpeed: 0.25,
      },
      highlights: () => {
        const map = new Map();
        const green = highlightOctet(closing.id, 2);
        const amber = highlightOctet(nonclosing.id, 3);
        if (green) for (const [r, c] of green) map.set(r, c);
        if (amber) for (const [r, c] of amber) map.set(r, c);
        return map;
      },
      verdict: garden.named_octets.map(o => ({
        id: o.id,
        status: o.published_status,
        abnormal: o.abnormal,
        signingExists: o.garden_signing_exists,
      })),
    },
    {
      id: 8,
      title: "The 168 ignite",
      dataset: "s8",
      caption: `Gold, used here and nowhere else: the ${s8.abnormal_right_slices.length} `
        + `left-right coincident octets, ${s8.abnormal_right_slices.length * 8} vertices. `
        + `The normalizer of R8 in S8 has order ${nrm.normalizer_order}, and `
        + `${nrm.normalizer_order} / 8 = ${nrm.normalizer_cosets}: these cosets realize `
        + `${nrm.quotient_identification}. |GL(3,2)| = 168. `
        + `The open question this instrument ends on, not a claim: does any physical `
        + `invariant distinguish these 168 from the other 4,872? (open)`,
      targets: {
        fog: 0.55, right: 0.15, left: 0, moire: 0, foreshadow: 0, gold: 1, gather: 0.55,
        pointSize: 3.0, pointAlpha: 0.8, edgeAlpha: 0.03, scale: 1.3, rotorSpeed: 0.35,
      },
      highlights: null,
      verdict: null,
    },
  ];
}

/**
 * Ledger content. Two columns of equal weight: what the tool
 * verifiably reproduces from the checked-in JSON, and what it does
 * not claim. Wording for the caveats comes from the data files
 * themselves wherever they carry it.
 */
export function makeLedger({ s8, garden }) {
  const nrm = garden.normalizer;
  return {
    reproduced: [
      `${s8.metadata.vertex_count.toLocaleString()} vertices and `
        + `${s8.metadata.edge_count.toLocaleString()} edges, regular of degree `
        + `${s8.metadata.degree}, verified complete by the in-page dataset check.`,
      `${garden.cosets_scanned.toLocaleString()} right R8 cosets scanned, `
        + `${garden.signable_cosets.toLocaleString()} signable, `
        + `${garden.unsignable_cosets} unsignable.`,
      `Garden sign equation per coset: rank ${garden.rank_histogram[0].value}, `
        + `nullity ${garden.nullity_histogram[0].value}, `
        + `${garden.solution_count_per_coset.toLocaleString()} = 2^19 raw signings.`,
      `2^19 raw signings = 2^15 node-sign gauge x 2^4: sixteen honest dashing `
        + `classes per coset, never 524,288 distinct physics.`,
      `${nrm.abnormal_cosets} left-right coincident (abnormal) octets; `
        + `normalizer order ${nrm.normalizer_order}; ${nrm.normalizer_order}/8 = 168 = |GL(3,2)|; `
        + `${nrm.quotient_identification}.`,
      `${garden.named_octets.length} named octets with their published statuses, `
        + `pinned from the source PDFs listed under provenance.`,
    ],
    notClaimed: [
      `The permutahedron-as-SUSY-weight-space reading is a research proposal, `
        + `not a theorem. Nothing rendered here proves it.`,
      `Quoting the scan boundary: "${garden.boundary}"`,
      `Quoting the scan conclusion: "${garden.conclusion}"`,
      `The vertex-pair number is "${s8.metadata.correlator_definition}". `
        + `It is not a holoraumy gadget and is not presented as one.`,
      `No new off-shell result is claimed anywhere in this instrument.`,
    ],
    provenance: s8.metadata.source.map(src => ({
      arxiv: `arXiv:${src.arxiv_id}v${src.version}`,
      sha256: src.pdf_sha256,
    })),
  };
}
