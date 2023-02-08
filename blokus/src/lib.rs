use fixedbitset::FixedBitSet;

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Color {
  Red = 0,
  Green = 1,
  Blue = 2,
  Yellow = 3,
}

struct State {
  // one bitset per player
  table: Vec<FixedBitSet>,
}
