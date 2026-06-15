use crate::code::DoublyEvenCode;
use nauty_pet::prelude::*;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

/// Build a colored bipartite graph representing a doubly-even code.
/// Column nodes (0..n) have color 0.
/// Nonzero codeword nodes have color = hamming_weight + 1.
/// Edges connect codewords to the columns where they have a 1 bit.
fn code_to_graph(code: &DoublyEvenCode) -> UnGraph<u32, ()> {
    let n = code.n;
    let codewords = code.all_codewords();

    let mut graph = UnGraph::new_undirected();

    // Add n column nodes with color 0
    for _ in 0..n {
        graph.add_node(0u32);
    }

    // Add nonzero codeword nodes colored by weight
    // Skip the zero codeword (it's in every code, has no edges)
    let nonzero_cws: Vec<u32> = codewords.into_iter().filter(|&c| c != 0).collect();
    for &cw in &nonzero_cws {
        graph.add_node(cw.count_ones() + 1); // +1 to avoid collision with column color 0
    }

    // Add edges: codeword node <-> column node
    for (cw_idx, &cw) in nonzero_cws.iter().enumerate() {
        let cw_node = NodeIndex::new(n + cw_idx);
        for col in 0..n {
            if cw & (1u32 << col) != 0 {
                graph.add_edge(cw_node, NodeIndex::new(col), ());
            }
        }
    }

    graph
}

/// Compute an exact canonical form key for a doubly-even code.
/// Two codes produce the same key iff they are equivalent under column permutation.
/// Returns a serialized representation of the nauty canonical graph.
pub fn exact_canonical_key(code: &DoublyEvenCode) -> Vec<u64> {
    if code.k() == 0 {
        return vec![];
    }

    let graph = code_to_graph(code);
    let canon: CanonGraph<u32, (), petgraph::Undirected, _> = CanonGraph::from(graph);

    // Serialize: node weights followed by sorted edge pairs
    // This gives a unique hashable key for each canonical graph
    let mut key = Vec::new();

    // Encode node count and edge count as header
    key.push(canon.node_count() as u64);
    key.push(canon.edge_count() as u64);

    // Encode node weights (in canonical order)
    for idx in canon.node_indices() {
        key.push(*canon.node_weight(idx).unwrap() as u64);
    }

    // Encode sorted edge list
    let mut edges: Vec<(usize, usize)> = canon
        .edge_references()
        .map(|e| {
            let a = e.source().index();
            let b = e.target().index();
            if a <= b {
                (a, b)
            } else {
                (b, a)
            }
        })
        .collect();
    edges.sort();
    for (a, b) in edges {
        key.push(((a as u64) << 32) | (b as u64));
    }

    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::deduplicate;
    use crate::code::enumerate_codes;

    #[test]
    fn nauty_matches_exact_n4() {
        // At N=4, exact enumeration gives known number of classes
        let codes = enumerate_codes(4);
        let exact_classes = deduplicate(codes.clone());

        // Now deduplicate using nauty
        let mut nauty_set: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
        for code in &codes {
            if code.k() > 0 {
                let key = exact_canonical_key(code);
                nauty_set.insert(key);
            }
        }

        let exact_nontrivial = exact_classes.iter().filter(|c| c.k() > 0).count();
        assert_eq!(
            nauty_set.len(),
            exact_nontrivial,
            "N=4: nauty found {} classes but exact enumeration found {}",
            nauty_set.len(),
            exact_nontrivial
        );
    }

    #[test]
    fn nauty_matches_exact_n8() {
        let codes = enumerate_codes(8);
        let exact_classes = deduplicate(codes.clone());

        let mut nauty_set: std::collections::HashSet<Vec<u64>> = std::collections::HashSet::new();
        for code in &codes {
            if code.k() > 0 {
                let key = exact_canonical_key(code);
                nauty_set.insert(key);
            }
        }

        let exact_nontrivial = exact_classes.iter().filter(|c| c.k() > 0).count();
        assert_eq!(
            nauty_set.len(),
            exact_nontrivial,
            "N=8: nauty found {} classes but exact enumeration found {}",
            nauty_set.len(),
            exact_nontrivial
        );
    }

    #[test]
    fn equivalent_codes_same_key() {
        // The [4,1,4] code {0000, 1111}
        // Permuting columns should give the same canonical key
        let c1 = DoublyEvenCode::new(4, vec![0b1111]);
        let c2 = DoublyEvenCode::new(4, vec![0b1111]); // same code, trivial perm
        assert_eq!(exact_canonical_key(&c1), exact_canonical_key(&c2));
    }

    #[test]
    fn inequivalent_codes_different_key() {
        // Two codes with different weight enumerators must be inequivalent
        let c1 = DoublyEvenCode::new(8, vec![0b11110000]); // [8,1,4]
        let c2 = DoublyEvenCode::new(8, vec![0b11111111]); // [8,1,8]
        assert_ne!(exact_canonical_key(&c1), exact_canonical_key(&c2));
    }
}
