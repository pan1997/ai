//! Dirichlet exploration noise utilities for root node injection in self-play RL pipelines.
//!
//! In AlphaZero and MuZero, exploration during self-play training is driven by adding
//! Dirichlet noise exclusively to the policy priors at the root node:
//!
//! $$P'(s_{\text{root}}, a) = (1 - \varepsilon) \cdot P(s_{\text{root}}, a) + \varepsilon \cdot \eta_a, \quad \boldsymbol{\eta} \sim \text{Dir}(\alpha \cdot \mathbf{1})$$
//!
//! Where:
//! - $\alpha > 0$ controls the concentration parameter (typically scaled inversely with the average legal move count, e.g. 0.3 for Chess, 0.03 for Go, 0.15 for Connect 4).
//! - $\varepsilon \in [0, 1]$ is the exploration fraction (typically 0.25).
//! - Internal tree nodes never receive Dirichlet noise during search.

use crate::tree_store::{EdgeStatsStore, NodeId, NodeStatus, PriorStore, TreeStore};
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};

/// Mixes Dirichlet exploration noise into a slice of policy prior probabilities in-place.
///
/// If `priors` is empty or has length 1, no noise is injected.
///
/// # Panics
///
/// Panics if `alpha <= 0.0` or `epsilon` is outside `[0.0, 1.0]`.
pub fn add_dirichlet_noise<R: Rng + ?Sized>(
    priors: &mut [f32],
    alpha: f32,
    epsilon: f32,
    rng: &mut R,
) {
    if priors.len() <= 1 {
        return;
    }
    assert!(alpha > 0.0, "add_dirichlet_noise: alpha must be strictly positive");
    assert!(
        (0.0..=1.0).contains(&epsilon),
        "add_dirichlet_noise: epsilon must be in range [0.0, 1.0]"
    );

    let dirichlet = Dirichlet::new_with_size(alpha, priors.len())
        .expect("add_dirichlet_noise: failed to construct Dirichlet distribution");
    let noise: Vec<f32> = dirichlet.sample(rng);

    for (p, &n) in priors.iter_mut().zip(noise.iter()) {
        *p = (1.0 - epsilon) * (*p) + epsilon * n;
    }
}

/// Injects Dirichlet exploration noise directly into the root node edges of a [`TreeStore`].
///
/// Queries child edge priors from `tree.stats`, mixes Dirichlet noise, and updates the stored priors.
///
/// # Panics
///
/// Panics if `root` is invalid or unexpanded.
pub fn add_root_dirichlet_noise<Action, Reward, Stats, R>(
    tree: &mut TreeStore<Action, Reward, Stats>,
    root: NodeId,
    alpha: f32,
    epsilon: f32,
    rng: &mut R,
) where
    Stats: EdgeStatsStore + PriorStore,
    R: Rng + ?Sized,
{
    assert!(root.is_valid(), "add_root_dirichlet_noise: root node handle is invalid");
    assert_eq!(
        tree.node_status(root),
        NodeStatus::Expanded,
        "add_root_dirichlet_noise: root node must be expanded prior to injecting noise"
    );

    let count = tree.num_children(root);
    if count <= 1 {
        return;
    }

    let first_edge = tree.first_child_edge(root);
    let mut priors: Vec<f32> = (0..count)
        .map(|i| {
            let edge = crate::tree_store::EdgeId(first_edge.0 + i);
            tree.stats.prior(edge)
        })
        .collect();

    add_dirichlet_noise(&mut priors, alpha, epsilon, rng);

    for (i, &p) in priors.iter().enumerate() {
        let edge = crate::tree_store::EdgeId(first_edge.0 + i as u32);
        tree.stats.set_prior(edge, p);
    }
}
