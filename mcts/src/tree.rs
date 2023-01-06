


struct Node<A> {
  actions: Vec<ActionInfo<A>>,
  // observation
  children: Vec<Node<A>>,
}


struct ActionInfo<A> {
  action: A,
  action_reward: Average,
}