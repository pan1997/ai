import tempfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

from learner.models.convnet import AlphaZeroConvNet, export_alphazero_onnx
from learner.trainer import compute_alphazero_loss


def test_alphazero_convnet_shapes_and_gradient():
    batch_size = 4
    in_channels = 3
    height = 3
    width = 3
    action_dim = 9
    num_players = 2

    model = AlphaZeroConvNet(
        in_channels=in_channels,
        height=height,
        width=width,
        action_dim=action_dim,
        num_players=num_players,
        hidden_channels=32,
        num_res_blocks=1,
    )

    x = torch.randn(batch_size, in_channels, height, width)
    # Channel 2 is turn indicator (1.0 for P0, 0.0 for P1)
    x[:2, 2, :, :] = 1.0
    x[2:, 2, :, :] = 0.0

    policy_logits, values = model(x)

    assert policy_logits.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, num_players)
    # Tanh bounds
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)

    # Test all-heads forward inspection
    all_policies, all_values = model.forward_all_heads(x)
    assert all_policies.shape == (batch_size, num_players, action_dim)
    assert all_values.shape == (batch_size, num_players)

    # When channel 2 is 1.0 (P0), policy_logits matches P0 head
    torch.testing.assert_close(policy_logits[:2], all_policies[:2, 0])
    # When channel 2 is 0.0 (P1), policy_logits matches P1 head
    torch.testing.assert_close(policy_logits[2:], all_policies[2:, 1])

    # Test loss backprop
    target_policy = torch.softmax(torch.randn(batch_size, action_dim), dim=-1)
    target_value = torch.randn(batch_size, num_players).clamp(-1.0, 1.0)

    total_loss, p_loss, v_loss = compute_alphazero_loss(policy_logits, values, target_policy, target_value)
    assert total_loss.item() > 0.0

    total_loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def test_onnx_export_and_runtime_inference():
    in_channels = 3
    height = 3
    width = 3
    action_dim = 9
    num_players = 2

    model = AlphaZeroConvNet(
        in_channels=in_channels,
        height=height,
        width=width,
        action_dim=action_dim,
        num_players=num_players,
        hidden_channels=32,
        num_res_blocks=1,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        export_alphazero_onnx(model, onnx_path, (in_channels, height, width))

        assert onnx_path.exists()

        # Load with ONNX Runtime
        session = ort.InferenceSession(str(onnx_path))

        # Check inputs and outputs
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        assert input_names == ["state"]
        assert "policy" in output_names
        assert "value" in output_names

        # Evaluate dummy input
        dummy_state = np.random.randn(2, in_channels, height, width).astype(np.float32)
        outputs = session.run(None, {"state": dummy_state})

        policy_out, value_out = outputs[0], outputs[1]
        assert policy_out.shape == (2, action_dim)
        assert value_out.shape == (2, num_players)


def test_alphazero_convnet_connect4_dimensions():
    in_channels = 3
    height = 6
    width = 7
    action_dim = 7
    num_players = 2

    model = AlphaZeroConvNet(
        in_channels=in_channels,
        height=height,
        width=width,
        action_dim=action_dim,
        num_players=num_players,
        hidden_channels=32,
        num_res_blocks=2,
    )

    x = torch.randn(4, in_channels, height, width)
    p_logits, v_pred = model(x)
    assert p_logits.shape == (4, action_dim)
    assert v_pred.shape == (4, num_players)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "c4_model.onnx"
        export_alphazero_onnx(model, onnx_path, (in_channels, height, width))
        assert onnx_path.exists()

        session = ort.InferenceSession(str(onnx_path))
        dummy = np.random.randn(3, in_channels, height, width).astype(np.float32)
        outs = session.run(None, {"state": dummy})
        assert outs[0].shape == (3, 7)
        assert outs[1].shape == (3, 2)

