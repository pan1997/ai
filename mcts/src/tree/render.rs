use std::{fmt::Display, fs::File, io::Write};

use graphviz_rust::{
  attributes::{EdgeAttributes, NodeAttributes},
  dot_structures::{Edge as GEdge, EdgeTy, Graph, Id, Node as GNode, NodeId, Port, Stmt, Vertex},
  printer::{DotPrinter, PrinterContext},
};

use super::Node;

fn render_dv<A: Ord + Clone, O: Ord + Clone + Display>(
  node: &Node<A, O>,
  g: &mut Graph,
  theta: u32,
  depth: u32,
  count: &mut u32,
) -> NodeId {
  let node_id = *count;
  *count += 1;
  let leaf = depth == 0 || node.select_count() <= theta;
  let label = node_format(&node, leaf);
  let n = GNode::new(
    NodeId(Id::Plain(format!("{node_id}")), None),
    vec![
      NodeAttributes::label(label),
      NodeAttributes::shape(graphviz_rust::attributes::shape::plaintext),
    ],
  );

  g.add_stmt(Stmt::Node(n));

  if !leaf {
    let children = unsafe { &*node.children.get() };

    for (ix, o) in children.keys().enumerate() {
      let child_id = render_dv(&children[o], g, theta, depth - 1, count);

      let e = GEdge {
        ty: EdgeTy::Pair(
          Vertex::N(NodeId(
            Id::Plain(format!("{node_id}")),
            Some(Port(Some(Id::Plain(format!("{ix}"))), None)),
          )),
          Vertex::N(child_id),
        ),
        attributes: vec![EdgeAttributes::label(format!("\"{}\"", o.to_string()))],
      };
      g.add_stmt(Stmt::Edge(e));
    }
  }
  NodeId(Id::Plain(format!("{node_id}")), None)
}

pub fn render_tree<A: Ord + Clone, O: Ord + Clone + Display>(
  node: &Node<A, O>,
  theta: u32,
  depth: u32,
) -> Graph {
  let mut g = Graph::DiGraph {
    id: Id::Plain("".to_string()),
    strict: false,
    stmts: vec![],
  };
  let mut count = 0;
  render_dv(node, &mut g, theta, depth, &mut count);
  g
}

pub fn save_tree<A: Ord + Clone, O: Ord + Clone + Display>(
  node: &Node<A, O>,
  mut f: File,
  theta: u32,
  depth: u32,
) {
  let mut g = render_tree(node, theta, depth);
  let mut ctx = PrinterContext::default();
  write!(f, "{}", g.print(&mut ctx)).unwrap();
}

fn node_format<A: Ord + Clone, O: Ord + Clone + Display>(node: &Node<A, O>, leaf: bool) -> String {
  let children = unsafe { &*node.children.get() };
  let width = std::cmp::max(if leaf { 1 } else { children.len() }, 1);
  let out_row = if leaf || children.is_empty() {
    "".to_string()
  } else {
    let mut result = "<table border=\"0\" cellspacing=\"0\" cellborder=\"1\"><tr>".to_string();
    for (ix, o) in children.keys().enumerate() {
      result.push_str(&format!("<td port=\"{ix}\">\"{o}\"</td>"));
    }
    result.push_str("</tr></table>");
    result
  };
  let action_row = if leaf || node.actions.is_empty() {
    "".to_string()
  } else {
    let mut result = "<table border=\"0\" cellspacing=\"0\" cellborder=\"1\"><tr>".to_string();
    for (a, data) in node.actions.iter() {
      let select_count = data.select_count();
      let sample_count = data.value_of_next_state.sample_count();
      let v = data.action_value();
      result.push_str(&format!("<td>{v}:{select_count}:{sample_count}</td>"));
    }
    result.push_str("</tr></table>");
    result
  };
  format!(
    r#"<
<table border="0" cellspacing="0" cellborder="1">
<tr><td>{}</td></tr>
<tr><td>{:.4}</td></tr>
<tr><td>{action_row}</td></tr>
<tr><td>{out_row}</td></tr>
</table>
    >"#,
    node.select_count(),
    node.value.mean(),
  )
}
