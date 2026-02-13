///! This is a first stab at creating a rust implementation of a weighted phylogenetic tree
///!
///! Trees nodes will be generic so that they can be used to simulate different models
use std::{
    fmt::{self, Display},
    fs,
    io::{self, Write},
};

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
    fn new(node: N, children: Vec<(Self, Option<f64>)>) -> Self {
        Self { node, children }
    }

    /// Construct a new leaf - phylogeny without any children
    fn new_leaf(node: N) -> Self {
        Self {
            node,
            children: vec![],
        }
    }

    /// Join to phylogenies with a given parent
    fn join_with_parent(parent: N, l: Self, ld: f64, r: Self, rd: f64) -> Self {
        // this is for bottom up construction like NJ or UPGMA
        Self {
            node: parent,
            children: vec![(l, Some(ld)), (r, Some(rd))],
        }
    }
}

/// Representation of a node
#[derive(Debug, Serialize)]
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
#[derive(Debug, Serialize)]
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
        for (i, n) in self.leaves().enumerate() {
            writeln!(w, "{},{}", i, self.nodes[n].label)?;
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
