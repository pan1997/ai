#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AgentId(pub u32);

impl AgentId {
    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<u32> for AgentId {
    #[inline]
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for AgentId {
    #[inline]
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<AgentId> for usize {
    #[inline]
    fn from(agent: AgentId) -> Self {
        agent.as_usize()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Agent({})", self.0)
    }
}
