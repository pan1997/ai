use std::{fmt::Display, fs::File, io::Write};

use graphviz_rust::{
  attributes::{EdgeAttributes, NodeAttributes},
  dot_structures::{
    Edge as GEdge, EdgeTy, Graph, Id, Node as GNode, NodeId as GNid, Port, Stmt, Vertex,
  },
  printer::{DotPrinter, PrinterContext},
};

use super::Forest;
use crate::forest::Node;

fn render<A: Ord + Display, O: Ord + Display>(
  forest: &Forest<A, O>,
  node: &Node<A, O>,
  g: &mut Graph,
  theta: u32,
  depth: u32,
  count: &mut u32,
) -> GNid {
  let node_id = *count;
  *count += 1;
  let leaf = depth == 0 || node.select_count() <= theta;
  let label = node_format(&node, leaf);
  let n = GNode::new(
    GNid(Id::Plain(format!("{node_id}")), None),
    vec![
      NodeAttributes::label(label),
      NodeAttributes::shape(graphviz_rust::attributes::shape::plaintext),
    ],
  );
  g.add_stmt(Stmt::Node(n));

  if !leaf {
    let children = &node.children;

    for (ix, o) in children.keys().enumerate() {
      let child_id = render(forest, forest.node(children[o]), g, theta, depth - 1, count);

      let e = GEdge {
        ty: EdgeTy::Pair(
          Vertex::N(GNid(
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
  GNid(Id::Plain(format!("{node_id}")), None)
}

pub fn render_forest<A: Ord + Display, O: Ord + Display>(
  forest: &Forest<A, O>,
  theta: u32,
  depth: u32,
) -> Graph {
  let mut g = Graph::DiGraph {
    id: Id::Plain("".to_string()),
    strict: false,
    stmts: vec![],
  };
  let mut count = 0;
  for nid in forest.roots.iter() {
    render(forest, forest.node(*nid), &mut g, theta, depth, &mut count);
  }
  g
}

pub fn save<A: Ord + Display, O: Ord + Display>(
  forest: &Forest<A, O>,
  mut f: File,
  theta: u32,
  depth: u32,
) {
  let g = render_forest(forest, theta, depth);
  let mut ctx = PrinterContext::default();
  write!(f, "{}", g.print(&mut ctx)).unwrap();
}

fn node_format<A: Ord + Display, O: Ord + Display>(node: &Node<A, O>, leaf: bool) -> String {
  let children = &node.children;
  let out_row = if leaf || children.is_empty() {
    "".to_string()
  } else {
    let mut result =
      "<table bgcolor=\"tomato\" border=\"0\" cellspacing=\"0\" cellborder=\"1\"><tr>".to_string();
    for (ix, o) in children.keys().enumerate() {
      result.push_str(&format!("<td port=\"{ix}\">{o}</td>"));
    }
    result.push_str("</tr></table>");
    result
  };
  let action_row = if leaf || node.actions.is_empty() {
    "".to_string()
  } else {
    let mut result =
      "<table bgcolor=\"gold\" border=\"0\" cellspacing=\"0\" cellborder=\"1\"><tr>".to_string();
    for (a, data) in node.actions.iter() {
      let ac = data.select_count();
      let ss = data.static_policy_score;
      result.push_str(&format!(
        "<td>{a}<BR/>{ss:.3}<BR/>{ac}</td>"
      ));
    }
    result.push_str("</tr></table>");
    result
  };
  format!(
    r#"<
<table border="0" cellspacing="0" cellborder="1">
<tr><td>{}</td></tr>
<tr><td>{:.4}, {}</td></tr>
<tr><td>{action_row}</td></tr>
<tr><td>{out_row}</td></tr>
</table>
    >"#,
    node.select_count(),
    node.value.value(),
    node.value.count()
  )
}
