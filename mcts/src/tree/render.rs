use std::{fs::File, fmt::Display};
use std::io::Write;

use graphviz_rust::attributes::NodeAttributes;
use graphviz_rust::{dot_structures::{Graph, Node as GNode, Edge as GEdge, NodeId, Id, Stmt, EdgeTy, Vertex, Attribute}, printer::{PrinterContext, DotPrinter}, attributes::EdgeAttributes};
use super::Node;

fn render_dv<A: Ord + Clone, O: Ord + Clone + Display>(
  node: &Node<A, O>,
  g: &mut Graph,
  theta: u32,
  depth: u32,
  count: &mut u32,
) -> NodeId {
  let id = NodeId(Id::Plain(format!("{}", count)), None);
  let n = GNode::new(id.clone(), vec![
    NodeAttributes::label(
      format!("<{}<BR ALIGN=\"LEFT\"/>{:.4}>", node.select_count(), node.value.mean())
    ),
    //NodeAttributes::shape(graphviz_rust::attributes::shape::plaintext)
  ]);
  *count += 1;

  g.add_stmt(Stmt::Node(n));

  if depth > 0 && node.select_count() > theta {
    let children = unsafe {&*node.children.get()};

    for o in children.keys() {
      let child_id  = render_dv(&children[o], g, theta, depth - 1, count);

      let e = GEdge {
        ty: EdgeTy::Pair(Vertex::N(id.clone()), Vertex::N(child_id)),
        attributes: vec![
          EdgeAttributes::label(format!("\"{}\"", o.to_string())),
        ],
      };
      g.add_stmt(Stmt::Edge(e));
    }
  }
  id
}


pub fn render_tree<A: Ord + Clone, O: Ord + Clone + Display>(
  node: &Node<A, O>
) -> Graph {
  let mut g = Graph::DiGraph { id: Id::Plain("".to_string()), strict: false, stmts: vec![] };
  let mut count = 0;
  render_dv(node, &mut g, 0, 4, &mut count);
  g
}

pub fn save_tree<A: Ord + Clone, O: Ord + Clone + Display>(
  node: &Node<A, O>,
  mut f: File
) {
  let mut g = render_tree(node);
  let mut ctx = PrinterContext::default();
  write!(f, "{}", g.print(&mut ctx)).unwrap();
}
