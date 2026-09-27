import tempfile
from pathlib import Path
import torch
import torch.optim as optim

from learner.models.tabular import TabularAlphaZero


def test_tabular_alphazero_forward_and_learn():
    model = TabularAlphaZero(num_states=10, num_actions=4, num_players=1)

    state_ids = torch.tensor([0, 3, 7], dtype=torch.long)
    logits, values = model(state_ids)

    assert logits.shape == (3, 4)
    assert values.shape == (3, 1)

    # Initial weights should be zero
    assert torch.all(logits == 0.0)
    assert torch.all(values == 0.0)

    # Test training step
    target_logits = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
    ])
    target_values = torch.tensor([[1.0], [-1.0], [0.5]])

    optimizer = optim.Adam(model.parameters(), lr=0.5)
    for _ in range(100):
        optimizer.zero_grad()
        l, v = model(state_ids)
        loss = ((l - target_logits) ** 2).sum() + ((v - target_values) ** 2).sum()
        loss.backward()
        optimizer.step()

    l, v = model(state_ids)
    assert torch.allclose(l, target_logits, atol=0.2)
    assert torch.allclose(v, target_values, atol=0.2)


def test_tabular_binary_serialization_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "tabular_model.bin"

        model = TabularAlphaZero(num_states=5, num_actions=3, num_players=2)
        model.policy_table.data.copy_(torch.randn(5, 3))
        model.value_table.data.copy_(torch.randn(5, 2))

        model.save_binary(save_path)
        assert save_path.exists()

        loaded_model = TabularAlphaZero.load_binary(save_path)
        assert loaded_model.num_states == 5
        assert loaded_model.num_actions == 3
        assert loaded_model.num_players == 2

        assert torch.allclose(model.policy_table, loaded_model.policy_table)
        assert torch.allclose(model.value_table, loaded_model.value_table)
