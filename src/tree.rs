//! This is a first stab at creating a rust implementation of a weighted phylogenetic tree
//!
//! Trees nodes will be generic so that they can be used to simulate different models
use std::{
    fmt::{self, Display},
    fs,
    io::{self, Write},
};

use rand::{Rng, thread_rng};
use serde::Serialize;

/// A simple recursive style tree structure for tree building algorithms like NJ and UPGMA
#[derive(Serialize)]
pub struct Tree<N> {
    /// Data about the node - simple version is letting this be `usize` so that it is an id
    node: N,
    /// Set of children and their distances to the parent
    children: Vec<(Tree<N>, Option<f64>)>,
}

impl<N: Clone + Display> Display for Tree<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}:({}))",
            self.node,
            self.children
                .iter()
                .map(|(t, dist)| if !t.children.is_empty() {
                    match dist {
                        Some(dist) => format!("({t}:{dist})"),
                        None => format!("({t})"),
                    }
                } else {
                    match dist {
                        Some(dist) => format!("({}:{dist})", t.node),
                        None => format!("({})", t.node),
                    }
                })
                .collect::<String>()
        )
    }
}

impl<N: Clone> Tree<N> {
    /// Create a new phylogeny with no children
    pub fn new(node: N, children: Vec<(Self, Option<f64>)>) -> Self {
        Self { node, children }
    }

    /// Construct a new leaf - phylogeny without any children
    pub fn new_leaf(node: N) -> Self {
        Self {
            node,
            children: vec![],
        }
    }

    /// Join to phylogenies with a given parent
    pub fn join_with_parent(parent: N, l: Self, ld: f64, r: Self, rd: f64) -> Self {
        // this is for bottom up construction like NJ or UPGMA
        Self {
            node: parent,
            children: vec![(l, Some(ld)), (r, Some(rd))],
        }
    }
}

/// Representation of a node
#[derive(Debug, Serialize, Clone)]
pub struct Node<N, L> {
    pub data: N,
    pub label: L,
    parent: Option<usize>,
    children: Vec<(usize, f64)>,
}

impl<N, L> Node<N, L> {
    pub fn root(data: N, label: L) -> Self {
        Self {
            data,
            label,
            parent: None,
            children: vec![],
        }
    }

    pub fn update_label(&mut self, l: L) {
        self.label = l
    }
}

impl<N: Clone + Display, L: Display> Display for Node<N, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

/// Simulation data structure for building a phylogeny top down best for
/// simulation like tree construction as branching process
#[derive(Debug, Serialize, Clone)]
pub struct Phylogeny<N, L> {
    pub nodes: Vec<Node<N, L>>,
    root_length: f64,
    pub root: usize,
}

impl<N: Clone + Display, L: Display> Display for Phylogeny<N, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn write_node<N: fmt::Display, L: fmt::Display>(
            f: &mut fmt::Formatter<'_>,
            nodes: &[Node<N, L>],
            node_idx: usize,
        ) -> fmt::Result {
            let node = &nodes[node_idx];
            write!(f, "{}", node.data)?;
            if !node.children.is_empty() {
                write!(f, "(")?;
                for (i, (child_idx, dist)) in node.children.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write_node(f, nodes, *child_idx)?;
                    write!(f, ":{}", dist)?;
                }
                write!(f, ")")?;
            }
            Ok(())
        }

        write!(f, "(")?;
        write_node(f, &self.nodes, self.root)?;
        write!(f, ")")
    }
}

impl<N: Display + Clone, L: Display + Clone> Phylogeny<N, L> {
    pub fn write_csv<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(w, "parent,child,length")?;
        for (p, c, len) in self.edges() {
            writeln!(w, "{},{},{}", p, c, len)?;
        }
        Ok(())
    }

    pub fn write_tsv<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(w, "parent\tchild\tlength")?;
        for (p, c, len) in self.edges() {
            writeln!(w, "{}\t{}\t{}", p, c, len)?;
        }
        Ok(())
    }

    pub fn write_csv_vertex_labeling<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(w, "vertex,label")?;
        for (i, n) in self.nodes.iter().enumerate() {
            writeln!(w, "{},{}", i, n.label)?;
        }
        Ok(())
    }

    pub fn write_csv_leaf_labeling<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(w, "leaf,label")?;
        for n in self.leaves() {
            writeln!(w, "{},{}", n, self.nodes[n].label)?;
        }
        Ok(())
    }
}

impl<N: Serialize + Clone, L: Serialize + Clone> Phylogeny<N, L> {
    /// Dump json to fil
    pub fn json_dump(&self, fname: &str) -> io::Result<()> {
        fs::write(
            fname,
            serde_json::to_string_pretty(&self.to_tree()).unwrap(),
        )
    }
}

impl<N: Clone, L: Clone> Phylogeny<N, L> {
    /// Build a new phylogeny with a given root
    pub fn new(root: Node<N, L>, root_length: f64) -> Self {
        Self {
            nodes: vec![root],
            root_length,
            root: 0,
        }
    }

    pub fn copy(&self) -> Self {
        self.clone()
    }

    /// Get an iterator over leaves
    pub fn leaves(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n.children.is_empty() {
                true => Some(i),
                false => None,
            })
    }

    /// Add a new child to a given parent
    pub fn add_child(&mut self, parent: usize, data: N, label: L, dist: f64) -> usize {
        let id = self.nodes.len();
        // store the new node without any children
        self.nodes.push(Node {
            data,
            label,
            parent: Some(parent),
            children: vec![],
        });
        // add the new as a child
        self.nodes[parent].children.push((id, dist));
        id
    }

    pub fn to_tree(&self) -> Tree<N> {
        self.build_tree(self.root)
    }

    fn build_tree(&self, idx: usize) -> Tree<N> {
        let node = &self.nodes[idx];

        Tree {
            node: node.data.clone(),
            children: node
                .children
                .iter()
                .map(|&(child_idx, dist)| (self.build_tree(child_idx), Some(dist)))
                .collect(),
        }
    }

    /// Perturb `self` using `ops` random SPR opertions
    /// Clones `self` to create new tree
    pub fn perturb(&self, ops: usize) -> Self {
        let mut rng = thread_rng();
        let mut new = self.clone();

        let mut executed_ops = 0;

        while executed_ops < ops {
            let v = rng.gen_range(1..new.nodes.len());
            // get descendents of v
            let v_child: Vec<_> = new.descendents(v);

            let u = if let Some(p) = new.nodes[v].parent {
                p
            } else {
                continue;
            };

            // do not create new leaves
            if new.nodes[u].children.len() == 1 {
                continue;
            }

            let mut w = rng.gen_range(0..new.nodes.len());
            while w == v || w == u || v_child.contains(&w) {
                w = rng.gen_range(0..new.nodes.len());
            }

            // we have u -> v and w
            // make remove v as child to u
            // make v new child of w
            let v_edge = *new.nodes[u]
                .children
                .get(
                    new.nodes[u]
                        .children
                        .iter()
                        .position(|&(i, _)| i == v)
                        .unwrap(),
                )
                .unwrap();

            // remove v as child to parent
            let mut old_parent = new.nodes[u].clone();
            old_parent.children.remove(
                old_parent
                    .children
                    .iter()
                    .position(|&(i, _)| i == v)
                    .unwrap(),
            );
            new.nodes[u] = old_parent;

            // add v as child of w
            new.nodes[w].children.push(v_edge);
            new.nodes[v].parent = Some(w);

            executed_ops += 1;
        }

        new
    }

    fn descendents(&self, target: usize) -> Vec<usize> {
        let mut descendents = vec![];
        let mut queue = vec![target];

        while let Some(v) = queue.pop() {
            for &(c, _) in &self.nodes[v].children {
                queue.push(c);
            }
            descendents.push(v);
        }

        descendents
    }

    pub fn is_tree(&self) -> bool {
        let mut visited = vec![];
        let mut queue = vec![self.root];

        while visited.len() != self.nodes.len() {
            let v = match queue.pop() {
                Some(v) => v,
                None => {
                    println!("Disconnected");
                    return false;
                }
            };

            for (c, _) in &self.nodes[v].children {
                if visited.contains(c) {
                    println!("Cycle");
                    return false;
                }
                queue.push(*c);
            }
            visited.push(v);
        }

        true
    }
}

impl<N, L> Phylogeny<N, L> {
    pub fn edges(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .flat_map(|(parent_idx, node)| {
                node.children
                    .iter()
                    .map(move |&(child_idx, len)| (parent_idx, child_idx, len))
            })
    }
}

#[test]
fn build_phylogeny() {
    let root = Node::<usize, Option<usize>>::root(0, None);
    let mut tree = Phylogeny::new(root, 0.);

    let mut idx = 1;
    for _ in 0..2 {
        let leaves: Vec<_> = tree.leaves().collect();
        for leaf in leaves {
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
        }
    }

    println!("{:#?}", tree);
}

#[test]
fn build_tree() {
    let leaf1 = Tree::new_leaf(1);
    let leaf2 = Tree::new_leaf(2);
    let tree = Tree::join_with_parent(0, leaf1, 0.5, leaf2, 0.7);
    println!("{}", tree);
}

#[test]
fn test_perturb() {
    let root = Node::<usize, Option<usize>>::root(0, None);
    let mut tree = Phylogeny::new(root, 0.);

    let mut idx = 1;
    for _ in 0..2 {
        let leaves: Vec<_> = tree.leaves().collect();
        for leaf in leaves {
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
        }
    }

    println!("{:#?}", tree.perturb(2));
}

#[test]
fn is_tree() {
    let root = Node::<usize, Option<usize>>::root(0, None);
    let mut tree = Phylogeny::new(root, 0.);

    let mut idx = 1;
    for _ in 0..2 {
        let leaves: Vec<_> = tree.leaves().collect();
        for leaf in leaves {
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
        }
    }

    assert!(tree.is_tree())
}

#[test]
fn is_not_tree() {
    let root = Node::<usize, Option<usize>>::root(0, None);
    let mut tree = Phylogeny::new(root, 0.);

    let mut idx = 1;
    for _ in 0..2 {
        let leaves: Vec<_> = tree.leaves().collect();
        for leaf in leaves {
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
        }
    }

    // disconnect tree
    tree.nodes[1].children = vec![];

    assert!(!tree.is_tree())
}

#[test]
fn perturbed_remains_tree() {
    let root = Node::<usize, Option<usize>>::root(0, None);
    let mut tree = Phylogeny::new(root, 0.);

    let mut idx = 1;
    for _ in 0..10 {
        let leaves: Vec<_> = tree.leaves().collect();
        for leaf in leaves {
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
        }
    }

    let p_tree = tree.perturb(5);
    assert!(p_tree.is_tree())
}

#[test]
fn nodes_reachable() {
    let root = Node::<usize, Option<usize>>::root(0, None);
    let mut tree = Phylogeny::new(root, 0.);

    let mut idx = 1;
    for _ in 0..2 {
        let leaves: Vec<_> = tree.leaves().collect();
        for leaf in leaves {
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
            tree.add_child(leaf, idx, None, 0.5);
            idx += 1;
        }
    }

    println!("{:?}", tree.descendents(0));
    println!("{:?}", tree.descendents(1));
    println!("{:?}", tree.descendents(2));
    println!("{:?}", tree.descendents(4));
}
