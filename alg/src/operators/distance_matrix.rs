use crate::models::datacentre::{Datacentre, NodeID};
use rand::prelude::*;
use std::collections::{HashSet, VecDeque};

pub type DistanceMatrix = Vec<Vec<NodeID>>;

pub fn build_cache(graph: &Datacentre, num_nearest: usize) -> DistanceMatrix {
    let num_servers = graph.num_servers;

    // Cache with size num_servers x num_considered
    let mut cache: DistanceMatrix = new_dm(num_servers, num_nearest);

    for start in 0..num_servers {
        set_nearest(&mut cache, graph, start, num_nearest);
    }

    cache
}

pub fn build_cache_dtm(graph: &Datacentre, num_nearest: usize) -> DistanceMatrix {
    let num_servers = graph.num_servers;

    // Cache with size num_servers x num_considered
    let mut cache: DistanceMatrix = new_dm(num_servers, num_nearest);

    for start in 0..num_servers {
        set_nearest_dtm(&mut cache, graph, start, num_nearest);
    }

    cache
}

/**
 * Breadth first search is deterministic which would mean that the same servers would be
 * used repeatedly e.g. in a Fat Tree, the 'left-most' servers would be consistently selected.
 *
 * Instead we randomly select items for expansion from the current horizon. This can be performed
 * efficiently with just two vectors, one for the nodes on the current horizon and one for
 * nodes on the next horizon.  
 */
fn set_nearest(cache: &mut DistanceMatrix, dc: &Datacentre, start: NodeID, num_nearest: usize) {
    // If you know the branching rate of the topology this step can be made more efficient by
    // allocating memory in advance
    let mut current_horizon = vec![start];
    let mut next_horizon = Vec::new();

    let mut visited = HashSet::new();
    visited.insert(start);

    let mut num_seen = 0;

    let mut rng = thread_rng();

    while !current_horizon.is_empty() {
        // Choose random node to expand
        let rn = rng.gen_range(0..current_horizon.len());
        let node_id = current_horizon.swap_remove(rn);

        // If we've found a server, add it to the distance matrix
        if dc.is_server(node_id) {
            cache[start].push(node_id);

            num_seen = num_seen + 1;

            if num_seen >= num_nearest {
                return;
            }
        }

        let neighbours = &dc.graph[node_id];

        // Add neighbours to horizon
        for &neighbour in neighbours {
            if visited.contains(&neighbour) {
                continue;
            }

            next_horizon.push(neighbour);
            visited.insert(neighbour);
        }

        if current_horizon.is_empty() {
            current_horizon = next_horizon;
            next_horizon = Vec::new();
        }
    }
}

// The deterministic BFs for comparison
fn set_nearest_dtm(cache: &mut DistanceMatrix, dc: &Datacentre, start: NodeID, num_nearest: usize) {
    let mut current_horizon = VecDeque::new();
    current_horizon.push_back(start);

    let mut visited = HashSet::new();
    visited.insert(start);

    let mut num_seen = 0;

    while !current_horizon.is_empty() {
        // Choose random node to expand
        let node_id = current_horizon.pop_front().unwrap();

        // If we've found a server, add it to the distance matrix
        if dc.is_server(node_id) {
            cache[start].push(node_id);

            num_seen = num_seen + 1;

            if num_seen >= num_nearest {
                return;
            }
        }

        let neighbours = &dc.graph[node_id];

        // Add neighbours to horizon
        for &neighbour in neighbours {
            if visited.contains(&neighbour) {
                continue;
            }

            current_horizon.push_back(neighbour);
            visited.insert(neighbour);
        }
    }
}

fn new_dm(num_servers: usize, num_nearest: usize) -> DistanceMatrix {
    vec![Vec::with_capacity(num_nearest); num_servers]
}

// ----- Unit tests ---- //
#[cfg(test)]
mod tests {
    use crate::models::datacentre::FatTree;
    use crate::operators::distance_matrix::*;

    #[test]
    fn test_set_nearest() {
        let dc = FatTree::new(4);
        let num_nearest = 6;
        let num_servers = dc.num_servers;

        let mut dm = new_dm(num_servers, num_nearest);

        set_nearest(&mut dm, &dc, 6, num_nearest);

        // Check length
        assert_eq!(dm[6].len(), num_nearest);

        // Check content
        assert_eq!(dm[6][0], 6,);
        assert_eq!(dm[6][1], 7);

        // assert_eq!(dm[6][2].distance, 4);
        // assert_eq!(dm[6][3].distance, 4);
        assert!(vec![4, 5].contains(&dm[6][2]));
        assert!(vec![4, 5].contains(&dm[6][3]));
        assert_ne!(dm[6][2], dm[6][3]);

        // assert_eq!(dm[6][4].distance, 6);
        // assert_eq!(dm[6][5].distance, 6);

        // Check randomness
        let mut used_prev = Vec::new();
        for _ in 0..100 {
            dm[6] = vec![];
            set_nearest(&mut dm, &dc, 6, num_nearest);

            let node_id = dm[6][4];
            assert!(!vec![4, 5, 6, 7].contains(&node_id));
            used_prev.push(node_id);
        }

        used_prev.sort();
        used_prev.dedup();

        assert!(used_prev.len() > 1);
    }

    #[test]
    fn test_set_nearest_dtm() {
        let dc = FatTree::new(4);
        let num_nearest = 6;
        let num_servers = dc.num_servers;

        let mut dm = new_dm(num_servers, num_nearest);

        set_nearest_dtm(&mut dm, &dc, 0, num_nearest);
        set_nearest_dtm(&mut dm, &dc, 1, num_nearest);
        set_nearest_dtm(&mut dm, &dc, 2, num_nearest);

        // Check length
        assert_eq!(dm[0].len(), num_nearest);

        // Check content
        assert_eq!(dm[0][0], 0);
        assert_eq!(dm[0][1], 1);
        assert_eq!(dm[0][2], 2);
        assert_eq!(dm[0][3], 3);
        assert_eq!(dm[0][4], 4);
        assert_eq!(dm[0][5], 5);

        // Check length
        assert_eq!(dm[1].len(), num_nearest);

        // Check content
        assert_eq!(dm[1][1], 0);
        assert_eq!(dm[1][0], 1);
        assert_eq!(dm[1][2], 2);
        assert_eq!(dm[1][3], 3);
        assert_eq!(dm[1][4], 4);
        assert_eq!(dm[1][5], 5);

        // Check length
        assert_eq!(dm[2].len(), num_nearest);

        // Check content
        assert_eq!(dm[2][2], 0);
        assert_eq!(dm[2][3], 1);
        assert_eq!(dm[2][0], 2);
        assert_eq!(dm[2][1], 3);
        assert_eq!(dm[2][4], 4);
        assert_eq!(dm[2][5], 5);
    }

    #[test]
    fn test_build_cache() {
        let dc = FatTree::new(4);
        let num_nearest = 8;
        let num_servers = dc.num_servers;

        let dm = build_cache(&dc, num_nearest);

        // Check length
        assert_eq!(dm.len(), num_servers);

        for i in 0..num_servers {
            assert_eq!(dm[i].len(), num_nearest);
        }

        // Check content
        // Most of this is being checked in 'test_set_nearest'
        for i in 0..num_servers {
            assert_eq!(dm[i][0], i);
        }
    }
}
