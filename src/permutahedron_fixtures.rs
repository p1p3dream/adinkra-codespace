//! Primary-source fixtures for the S4 and S8 permutahedron calculations.
//!
//! The integer matrices in this module are Bruhat-distance correlators. They
//! are graph distances on a permutahedron. They are not holoraumy gadgets,
//! which are traces of products of holoraumy matrices.

/// A permutation in one-line notation.
pub type Permutation4 = [u8; 4];
/// A permutation in one-line notation.
pub type Permutation8 = [u8; 8];
/// A published 4 by 4 Bruhat-distance correlator block.
pub type Correlator4 = [[u8; 4]; 4];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceDocument {
    pub arxiv_id: &'static str,
    pub version: u8,
    pub local_pdf: &'static str,
    pub pdf_sha256: &'static str,
}

/// The four PDFs identified for the permutahedron and Adinkra review.
pub const PRIMARY_SOURCE_DOCUMENTS: [SourceDocument; 4] = [
    SourceDocument {
        arxiv_id: "2012.14015",
        version: 7,
        local_pdf: "01_2012.14015_A_note_on_exemplary_off-shell_constructions.pdf",
        pdf_sha256: "67606499dda53ff78e3d80ca405498d8d1e2fd060936605fac04804ca7cd9b47",
    },
    SourceDocument {
        arxiv_id: "2304.09830",
        version: 2,
        local_pdf: "02_2304.09830_N2_SUSY_and_the_7-Simplex.pdf",
        pdf_sha256: "d68c14ad31e824b52076bd4be078c6aec0167f0a622c55c4a9d4d2a6d2ef2a34",
    },
    SourceDocument {
        arxiv_id: "2311.06842",
        version: 1,
        local_pdf: "03_2311.06842_Unfolded_Adinkra_Properties_I.pdf",
        pdf_sha256: "346e5c3863f81d4afc8d0faf90a0f23c805062a21e9aebc1b32c96a1dd1b6e86",
    },
    SourceDocument {
        arxiv_id: "2012.13308",
        version: 6,
        local_pdf: "04_2012.13308_The_300_Correlators.pdf",
        pdf_sha256: "b75a70b4642ba62f973993d37da2310a374de3b0b3f86d18bcd81c11e08a7957",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceLocator {
    pub arxiv_id: &'static str,
    pub version: u8,
    pub pdf_pages: &'static str,
    pub equation_or_table: &'static str,
}

pub const S4_DICTIONARY_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2012.13308",
    version: 6,
    pdf_pages: "7",
    equation_or_table: "Eqs. (3.1) and (3.2)",
};
pub const S4_EDGE_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2012.13308",
    version: 6,
    pdf_pages: "12-14",
    equation_or_table: "Figs. 6-8 and the 36-edge statement in Sec. 4.2",
};
pub const S4_QUARTET_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2012.13308",
    version: 6,
    pdf_pages: "8 and 17-22",
    equation_or_table: "Fig. 2 and Tables 1-6",
};
pub const S4_CORRELATOR_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2012.13308",
    version: 6,
    pdf_pages: "17-26",
    equation_or_table: "Eqs. (5.2), (5.4), (5.6), (5.8), (5.10), (5.12), and Tables 7-21",
};
/// Exact locators aligned first with `S4_INTRA_CORRELATORS`, then with
/// `S4_INTER_CORRELATORS`.
pub const S4_CORRELATOR_BLOCK_SOURCES: [SourceLocator; 21] = [
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "17",
        equation_or_table: "Eq. (5.2), Table 1",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "18",
        equation_or_table: "Eq. (5.4), Table 2",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "19",
        equation_or_table: "Eq. (5.6), Table 3",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "20",
        equation_or_table: "Eq. (5.8), Table 4",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "21",
        equation_or_table: "Eq. (5.10), Table 5",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "22",
        equation_or_table: "Eq. (5.12), Table 6",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "23",
        equation_or_table: "Eq. (6.1), Table 7",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "23",
        equation_or_table: "Eq. (6.2), Table 8",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "23",
        equation_or_table: "Eq. (6.3), Table 9",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "24",
        equation_or_table: "Eq. (6.4), Table 10",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "24",
        equation_or_table: "Eq. (6.5), Table 11",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "24",
        equation_or_table: "Eq. (6.6), Table 12",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "24",
        equation_or_table: "Eq. (6.7), Table 13",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "25",
        equation_or_table: "Eq. (6.8), Table 14",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "25",
        equation_or_table: "Eq. (6.9), Table 15",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "25",
        equation_or_table: "Eq. (6.10), Table 16",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "25",
        equation_or_table: "Eq. (6.11), Table 17",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "26",
        equation_or_table: "Eq. (6.12), Table 18",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "26",
        equation_or_table: "Eq. (6.13), Table 19",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "26",
        equation_or_table: "Eq. (6.14), Table 20",
    },
    SourceLocator {
        arxiv_id: "2012.13308",
        version: 6,
        pdf_pages: "26",
        equation_or_table: "Eq. (6.15), Table 21",
    },
];
pub const S8_REPRESENTATION_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2012.14015",
    version: 7,
    pdf_pages: "8-11",
    equation_or_table: "one-line S8 addresses in Eqs. (5.1)-(5.6)",
};
pub const R8_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2304.09830",
    version: 2,
    pdf_pages: "19",
    equation_or_table: "Eq. (3.23)",
};
pub const S8_COMBINATORICS_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2304.09830",
    version: 2,
    pdf_pages: "4 and 9",
    equation_or_table: "Sec. 1 node, edge, face, and color counts; Sec. 2.2 octet count",
};
pub const S8_MAGIC_NUMBER_SOURCE: SourceLocator = SourceLocator {
    arxiv_id: "2304.09830",
    version: 2,
    pdf_pages: "21",
    equation_or_table: "Eqs. (3.27) and (3.28), including the stated S8 value",
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct S4DictionaryEntry {
    pub one_line: Permutation4,
    pub cycle_notation: &'static str,
}

/// The lexicographic S4 dictionary in Eqs. (3.1) and (3.2).
pub const S4_DICTIONARY: [S4DictionaryEntry; 24] = [
    S4DictionaryEntry {
        one_line: [1, 2, 3, 4],
        cycle_notation: "()",
    },
    S4DictionaryEntry {
        one_line: [1, 2, 4, 3],
        cycle_notation: "(34)",
    },
    S4DictionaryEntry {
        one_line: [1, 3, 2, 4],
        cycle_notation: "(23)",
    },
    S4DictionaryEntry {
        one_line: [1, 3, 4, 2],
        cycle_notation: "(234)",
    },
    S4DictionaryEntry {
        one_line: [1, 4, 2, 3],
        cycle_notation: "(243)",
    },
    S4DictionaryEntry {
        one_line: [1, 4, 3, 2],
        cycle_notation: "(24)",
    },
    S4DictionaryEntry {
        one_line: [2, 1, 3, 4],
        cycle_notation: "(12)",
    },
    S4DictionaryEntry {
        one_line: [2, 1, 4, 3],
        cycle_notation: "(12)(34)",
    },
    S4DictionaryEntry {
        one_line: [2, 3, 1, 4],
        cycle_notation: "(123)",
    },
    S4DictionaryEntry {
        one_line: [2, 3, 4, 1],
        cycle_notation: "(1234)",
    },
    S4DictionaryEntry {
        one_line: [2, 4, 1, 3],
        cycle_notation: "(1243)",
    },
    S4DictionaryEntry {
        one_line: [2, 4, 3, 1],
        cycle_notation: "(124)",
    },
    S4DictionaryEntry {
        one_line: [3, 1, 2, 4],
        cycle_notation: "(132)",
    },
    S4DictionaryEntry {
        one_line: [3, 1, 4, 2],
        cycle_notation: "(1342)",
    },
    S4DictionaryEntry {
        one_line: [3, 2, 1, 4],
        cycle_notation: "(13)",
    },
    S4DictionaryEntry {
        one_line: [3, 2, 4, 1],
        cycle_notation: "(134)",
    },
    S4DictionaryEntry {
        one_line: [3, 4, 1, 2],
        cycle_notation: "(13)(24)",
    },
    S4DictionaryEntry {
        one_line: [3, 4, 2, 1],
        cycle_notation: "(1324)",
    },
    S4DictionaryEntry {
        one_line: [4, 1, 2, 3],
        cycle_notation: "(1432)",
    },
    S4DictionaryEntry {
        one_line: [4, 1, 3, 2],
        cycle_notation: "(142)",
    },
    S4DictionaryEntry {
        one_line: [4, 2, 1, 3],
        cycle_notation: "(143)",
    },
    S4DictionaryEntry {
        one_line: [4, 2, 3, 1],
        cycle_notation: "(14)",
    },
    S4DictionaryEntry {
        one_line: [4, 3, 1, 2],
        cycle_notation: "(1423)",
    },
    S4DictionaryEntry {
        one_line: [4, 3, 2, 1],
        cycle_notation: "(14)(23)",
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdjacentTransposition {
    /// Swap positions 1 and 2.
    S1,
    /// Swap positions 2 and 3.
    S2,
    /// Swap positions 3 and 4.
    S3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeFixture {
    /// Index in `S4_DICTIONARY`.
    pub from: u8,
    /// Index in `S4_DICTIONARY`.
    pub to: u8,
    pub generator: AdjacentTransposition,
}

/// The 36 labeled edges of the weak-right-Bruhat S4 permutahedron.
pub const S4_LABELED_EDGES: [EdgeFixture; 36] = [
    EdgeFixture {
        from: 0,
        to: 6,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 0,
        to: 2,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 0,
        to: 1,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 1,
        to: 7,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 1,
        to: 4,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 2,
        to: 12,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 2,
        to: 3,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 3,
        to: 13,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 3,
        to: 5,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 4,
        to: 18,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 4,
        to: 5,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 5,
        to: 19,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 6,
        to: 8,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 6,
        to: 7,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 7,
        to: 10,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 8,
        to: 14,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 8,
        to: 9,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 9,
        to: 15,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 9,
        to: 11,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 10,
        to: 20,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 10,
        to: 11,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 11,
        to: 21,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 12,
        to: 14,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 12,
        to: 13,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 13,
        to: 16,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 14,
        to: 15,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 15,
        to: 17,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 16,
        to: 22,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 16,
        to: 17,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 17,
        to: 23,
        generator: AdjacentTransposition::S1,
    },
    EdgeFixture {
        from: 18,
        to: 20,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 18,
        to: 19,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 19,
        to: 22,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 20,
        to: 21,
        generator: AdjacentTransposition::S3,
    },
    EdgeFixture {
        from: 21,
        to: 23,
        generator: AdjacentTransposition::S2,
    },
    EdgeFixture {
        from: 22,
        to: 23,
        generator: AdjacentTransposition::S3,
    },
];

/// Six ordered SUSY quartets, P[1] through P[6].
pub const S4_ORDERED_QUARTETS: [[Permutation4; 4]; 6] = [
    [[1, 4, 2, 3], [2, 3, 1, 4], [3, 2, 4, 1], [4, 1, 3, 2]],
    [[1, 3, 4, 2], [2, 4, 3, 1], [3, 1, 2, 4], [4, 2, 1, 3]],
    [[1, 3, 2, 4], [2, 4, 1, 3], [3, 1, 4, 2], [4, 2, 3, 1]],
    [[1, 4, 3, 2], [2, 3, 4, 1], [3, 2, 1, 4], [4, 1, 2, 3]],
    [[1, 2, 4, 3], [2, 1, 3, 4], [3, 4, 2, 1], [4, 3, 1, 2]],
    [[1, 2, 3, 4], [2, 1, 4, 3], [3, 4, 1, 2], [4, 3, 2, 1]],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BruhatCorrelatorBlock {
    /// One-based P[A] label for the block's rows.
    pub row_quartet: u8,
    /// One-based P[B] label for the block's columns.
    pub column_quartet: u8,
    pub values: Correlator4,
}

/// The six published intra-quartet Bruhat-distance correlators.
pub const S4_INTRA_CORRELATORS: [BruhatCorrelatorBlock; 6] = [
    BruhatCorrelatorBlock {
        row_quartet: 1,
        column_quartet: 1,
        values: [[0, 4, 6, 2], [4, 0, 2, 6], [6, 2, 0, 4], [2, 6, 4, 0]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 2,
        column_quartet: 2,
        values: [[0, 6, 2, 4], [6, 0, 4, 2], [2, 4, 0, 6], [4, 2, 6, 0]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 3,
        column_quartet: 3,
        values: [[0, 4, 2, 6], [4, 0, 6, 2], [2, 6, 0, 4], [6, 2, 4, 0]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 4,
        column_quartet: 4,
        values: [[0, 6, 4, 2], [6, 0, 2, 4], [4, 2, 0, 6], [2, 4, 6, 0]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 5,
        column_quartet: 5,
        values: [[0, 2, 6, 4], [2, 0, 4, 6], [6, 4, 0, 2], [4, 6, 2, 0]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 6,
        column_quartet: 6,
        values: [[0, 2, 4, 6], [2, 0, 6, 4], [4, 6, 0, 2], [6, 4, 2, 0]],
    },
];

/// The 15 oriented inter-quartet blocks as printed in Tables 7-21.
/// Rows and columns retain the paper's orientation; the omitted reverse block
/// is the transpose.
pub const S4_INTER_CORRELATORS: [BruhatCorrelatorBlock; 15] = [
    BruhatCorrelatorBlock {
        row_quartet: 6,
        column_quartet: 1,
        values: [[2, 2, 4, 4], [2, 2, 4, 4], [4, 4, 2, 2], [4, 4, 2, 2]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 6,
        column_quartet: 2,
        values: [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 6,
        column_quartet: 3,
        values: [[1, 3, 3, 5], [3, 1, 5, 3], [3, 5, 1, 3], [5, 3, 3, 1]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 6,
        column_quartet: 4,
        values: [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 6,
        column_quartet: 5,
        values: [[1, 1, 5, 5], [1, 1, 5, 5], [5, 5, 1, 1], [5, 5, 1, 1]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 5,
        column_quartet: 1,
        values: [[1, 3, 5, 3], [3, 1, 3, 5], [5, 3, 1, 3], [3, 5, 3, 1]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 5,
        column_quartet: 2,
        values: [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 5,
        column_quartet: 3,
        values: [[2, 2, 4, 4], [2, 2, 4, 4], [4, 4, 2, 2], [4, 4, 2, 2]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 5,
        column_quartet: 4,
        values: [[2, 4, 4, 2], [4, 2, 2, 4], [4, 2, 2, 4], [2, 4, 4, 2]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 4,
        column_quartet: 1,
        values: [[1, 5, 5, 1], [5, 1, 1, 5], [5, 1, 1, 5], [1, 5, 5, 1]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 4,
        column_quartet: 2,
        values: [[1, 5, 3, 3], [5, 1, 3, 3], [3, 3, 1, 5], [3, 3, 5, 1]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 4,
        column_quartet: 3,
        values: [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 3,
        column_quartet: 1,
        values: [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 3,
        column_quartet: 2,
        values: [[1, 5, 1, 5], [5, 1, 5, 1], [1, 5, 1, 5], [5, 1, 5, 1]],
    },
    BruhatCorrelatorBlock {
        row_quartet: 2,
        column_quartet: 1,
        values: [[2, 4, 4, 2], [4, 2, 2, 4], [4, 2, 2, 4], [2, 4, 4, 2]],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RepresentationOctet {
    pub label: &'static str,
    /// Unsigned permutation factors, ordered by L1 through L8.
    pub permutations: [Permutation8; 8],
}

/// Unsigned S8 permutation octets for the six paired representations.
pub const S8_REPRESENTATION_OCTETS: [RepresentationOctet; 6] = [
    RepresentationOctet {
        label: "CC",
        permutations: [
            [1, 4, 2, 3, 5, 8, 6, 7],
            [2, 3, 1, 4, 6, 7, 5, 8],
            [3, 2, 4, 1, 7, 6, 8, 5],
            [4, 1, 3, 2, 8, 5, 7, 6],
            [5, 8, 6, 7, 1, 4, 2, 3],
            [6, 7, 5, 8, 2, 3, 1, 4],
            [7, 6, 8, 5, 3, 2, 4, 1],
            [8, 5, 7, 6, 4, 1, 3, 2],
        ],
    },
    RepresentationOctet {
        label: "CT",
        permutations: [
            [1, 4, 2, 3, 5, 7, 8, 6],
            [2, 3, 1, 4, 6, 8, 7, 5],
            [3, 2, 4, 1, 7, 5, 6, 8],
            [4, 1, 3, 2, 8, 6, 5, 7],
            [5, 8, 6, 7, 1, 3, 4, 2],
            [6, 7, 5, 8, 2, 4, 3, 1],
            [7, 6, 8, 5, 3, 1, 2, 4],
            [8, 5, 7, 6, 4, 2, 1, 3],
        ],
    },
    RepresentationOctet {
        label: "CV",
        permutations: [
            [1, 4, 2, 3, 6, 8, 5, 7],
            [2, 3, 1, 4, 5, 7, 6, 8],
            [3, 2, 4, 1, 8, 6, 7, 5],
            [4, 1, 3, 2, 7, 5, 8, 6],
            [5, 8, 6, 7, 2, 4, 1, 3],
            [6, 7, 5, 8, 1, 3, 2, 4],
            [7, 6, 8, 5, 4, 2, 3, 1],
            [8, 5, 7, 6, 3, 1, 4, 2],
        ],
    },
    RepresentationOctet {
        label: "TT",
        permutations: [
            [1, 3, 4, 2, 5, 7, 8, 6],
            [2, 4, 3, 1, 6, 8, 7, 5],
            [3, 1, 2, 4, 7, 5, 6, 8],
            [4, 2, 1, 3, 8, 6, 5, 7],
            [5, 7, 8, 6, 1, 3, 4, 2],
            [6, 8, 7, 5, 2, 4, 3, 1],
            [7, 5, 6, 8, 3, 1, 2, 4],
            [8, 6, 5, 7, 4, 2, 1, 3],
        ],
    },
    RepresentationOctet {
        label: "TV",
        permutations: [
            [1, 3, 4, 2, 6, 8, 5, 7],
            [2, 4, 3, 1, 5, 7, 6, 8],
            [3, 1, 2, 4, 8, 6, 7, 5],
            [4, 2, 1, 3, 7, 5, 8, 6],
            [5, 7, 8, 6, 2, 4, 1, 3],
            [6, 8, 7, 5, 1, 3, 2, 4],
            [7, 5, 6, 8, 4, 2, 3, 1],
            [8, 6, 5, 7, 3, 1, 4, 2],
        ],
    },
    RepresentationOctet {
        label: "VV",
        permutations: [
            [2, 4, 1, 3, 6, 8, 5, 7],
            [1, 3, 2, 4, 5, 7, 6, 8],
            [4, 2, 3, 1, 8, 6, 7, 5],
            [3, 1, 4, 2, 7, 5, 8, 6],
            [6, 8, 5, 7, 2, 4, 1, 3],
            [5, 7, 6, 8, 1, 3, 2, 4],
            [8, 6, 7, 5, 4, 2, 3, 1],
            [7, 5, 8, 6, 3, 1, 4, 2],
        ],
    },
];

/// The Rana subgroup R8, also called the unsigned Diadem(8) octet.
pub const R8_DIADEM_OCTET: [Permutation8; 8] = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [2, 1, 4, 3, 6, 5, 8, 7],
    [3, 4, 1, 2, 7, 8, 5, 6],
    [4, 3, 2, 1, 8, 7, 6, 5],
    [5, 6, 7, 8, 1, 2, 3, 4],
    [6, 5, 8, 7, 2, 1, 4, 3],
    [7, 8, 5, 6, 3, 4, 1, 2],
    [8, 7, 6, 5, 4, 3, 2, 1],
];

pub const S8_VERTEX_COUNT: usize = 40_320;
pub const S8_EDGE_COUNT: usize = 141_120;
pub const S8_FACE_COUNT: usize = 191_520;
pub const S8_COLOR_COUNT: usize = 8;
pub const S8_OCTET_COUNT: usize = 5_040;
pub const S8_MAGIC_NUMBER: usize = 112;
pub const S4_SYMMETRIC_CORRELATOR_ENTRY_COUNT: usize = 300;

fn dictionary_index(permutation: Permutation4) -> Option<usize> {
    S4_DICTIONARY
        .iter()
        .position(|entry| entry.one_line == permutation)
}

/// Reconstruct the full symmetric 24 by 24 Bruhat-distance matrix from the
/// 21 published blocks. This function does not compute a holoraumy gadget.
pub fn reconstruct_s4_bruhat_correlator() -> Result<[[u8; 24]; 24], &'static str> {
    const UNSET: u8 = u8::MAX;
    let mut result = [[UNSET; 24]; 24];

    for block in S4_INTRA_CORRELATORS
        .iter()
        .chain(S4_INTER_CORRELATORS.iter())
    {
        if !(1..=6).contains(&block.row_quartet) || !(1..=6).contains(&block.column_quartet) {
            return Err("quartet label outside 1 through 6");
        }
        let row_quartet = &S4_ORDERED_QUARTETS[(block.row_quartet - 1) as usize];
        let column_quartet = &S4_ORDERED_QUARTETS[(block.column_quartet - 1) as usize];
        for (local_row, permutation_row) in row_quartet.iter().enumerate() {
            let row = dictionary_index(*permutation_row).ok_or("unknown row permutation")?;
            for (local_column, permutation_column) in column_quartet.iter().enumerate() {
                let column =
                    dictionary_index(*permutation_column).ok_or("unknown column permutation")?;
                let value = block.values[local_row][local_column];
                if result[row][column] != UNSET && result[row][column] != value {
                    return Err("conflicting published block entries");
                }
                result[row][column] = value;
                if block.row_quartet != block.column_quartet {
                    if result[column][row] != UNSET && result[column][row] != value {
                        return Err("conflicting transposed block entries");
                    }
                    result[column][row] = value;
                }
            }
        }
    }

    if result.iter().flatten().any(|value| *value == UNSET) {
        return Err("published blocks do not cover the full matrix");
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, VecDeque};

    #[test]
    fn s4_dictionary_is_lexicographic_and_unique() {
        let permutations: Vec<_> = S4_DICTIONARY.iter().map(|entry| entry.one_line).collect();
        assert!(permutations.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            permutations.iter().copied().collect::<HashSet<_>>().len(),
            24
        );
    }

    #[test]
    fn six_quartets_partition_s4() {
        let members: HashSet<_> = S4_ORDERED_QUARTETS.iter().flatten().copied().collect();
        let dictionary: HashSet<_> = S4_DICTIONARY.iter().map(|entry| entry.one_line).collect();
        assert_eq!(members.len(), 24);
        assert_eq!(members, dictionary);
        assert!(
            S4_ORDERED_QUARTETS
                .iter()
                .all(|quartet| quartet.windows(2).all(|pair| pair[0] < pair[1]))
        );
    }

    #[test]
    fn labeled_edges_are_the_36_adjacent_position_swaps() {
        let mut seen = HashSet::new();
        let mut degrees = [0_u8; 24];
        for edge in S4_LABELED_EDGES {
            let from = edge.from as usize;
            let to = edge.to as usize;
            assert!(from < to);
            assert!(seen.insert((from, to)));
            degrees[from] += 1;
            degrees[to] += 1;
            let mut expected = S4_DICTIONARY[from].one_line;
            let position = match edge.generator {
                AdjacentTransposition::S1 => 0,
                AdjacentTransposition::S2 => 1,
                AdjacentTransposition::S3 => 2,
            };
            expected.swap(position, position + 1);
            assert_eq!(expected, S4_DICTIONARY[to].one_line);
        }
        assert_eq!(seen.len(), 36);
        assert!(degrees.iter().all(|degree| *degree == 3));
    }

    #[test]
    fn published_blocks_cover_each_unordered_quartet_pair_once() {
        let mut pairs = HashSet::new();
        let mut represented_upper_entries = 0;
        for block in S4_INTRA_CORRELATORS
            .iter()
            .chain(S4_INTER_CORRELATORS.iter())
        {
            let pair = if block.row_quartet <= block.column_quartet {
                (block.row_quartet, block.column_quartet)
            } else {
                (block.column_quartet, block.row_quartet)
            };
            assert!(pairs.insert(pair));
            represented_upper_entries += if pair.0 == pair.1 { 10 } else { 16 };
        }
        assert_eq!(pairs.len(), 21);
        assert_eq!(
            represented_upper_entries,
            S4_SYMMETRIC_CORRELATOR_ENTRY_COUNT
        );
    }

    #[test]
    fn blocks_reconstruct_a_symmetric_24_by_24_matrix_with_300_entries() {
        let matrix = reconstruct_s4_bruhat_correlator().unwrap();
        let mut upper_triangle_entries = 0;
        for row in 0..24 {
            assert_eq!(matrix[row][row], 0);
            for column in row..24 {
                assert_eq!(matrix[row][column], matrix[column][row]);
                upper_triangle_entries += 1;
            }
        }
        assert_eq!(upper_triangle_entries, S4_SYMMETRIC_CORRELATOR_ENTRY_COUNT);
    }

    #[test]
    fn blocks_equal_shortest_paths_on_the_36_edge_fixture() {
        let matrix = reconstruct_s4_bruhat_correlator().unwrap();
        let mut adjacency = vec![Vec::new(); 24];
        for edge in S4_LABELED_EDGES {
            adjacency[edge.from as usize].push(edge.to as usize);
            adjacency[edge.to as usize].push(edge.from as usize);
        }
        for source in 0..24 {
            let mut distance = [u8::MAX; 24];
            distance[source] = 0;
            let mut queue = VecDeque::from([source]);
            while let Some(vertex) = queue.pop_front() {
                for &neighbor in &adjacency[vertex] {
                    if distance[neighbor] == u8::MAX {
                        distance[neighbor] = distance[vertex] + 1;
                        queue.push_back(neighbor);
                    }
                }
            }
            assert_eq!(matrix[source], distance);
        }
    }

    #[test]
    fn representation_octets_and_diadem_contain_permutations() {
        fn valid(permutation: &Permutation8) -> bool {
            let mut sorted = *permutation;
            sorted.sort_unstable();
            sorted == [1, 2, 3, 4, 5, 6, 7, 8]
        }
        assert!(
            S8_REPRESENTATION_OCTETS
                .iter()
                .flat_map(|octet| octet.permutations.iter())
                .all(valid)
        );
        assert!(R8_DIADEM_OCTET.iter().all(valid));
        assert_eq!(
            R8_DIADEM_OCTET
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len(),
            8
        );
    }

    #[test]
    fn r8_diadem_has_published_magic_number() {
        let identity = R8_DIADEM_OCTET[0];
        let coxeter_length_from_identity = |permutation: Permutation8| {
            let mut inversions = 0;
            for left in 0..8 {
                for right in left + 1..8 {
                    if permutation[left] > permutation[right] {
                        inversions += 1;
                    }
                }
            }
            inversions
        };
        assert_eq!(identity, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            R8_DIADEM_OCTET
                .iter()
                .copied()
                .map(coxeter_length_from_identity)
                .sum::<usize>(),
            S8_MAGIC_NUMBER
        );
    }
}
