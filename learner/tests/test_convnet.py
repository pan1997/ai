import tempfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

from learner.models.convnet import AlphaZeroConvNet, export_alphazero_onnx
from learner.trainer import compute_alphazero_loss


def test_alphazero_convnet_shapes_and_gradient():
    batch_size = 4
    in_channels = 2
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
    policy_logits, values = model(x)

    assert policy_logits.shape == (batch_size, action_dim)
    assert values.shape == (batch_size, num_players)
    # Tanh bounds
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)

    # Test loss backprop
    target_policy = torch.softmax(torch.randn(batch_size, action_dim), dim=-1)
    target_value = torch.randn(batch_size, num_players).clamp(-1.0, 1.0)

    total_loss, p_loss, v_loss = compute_alphazero_loss(policy_logits, values, target_policy, target_value)
    assert total_loss.item() > 0.0

    total_loss.backward()
    for param in model.parameters():
        assert param.grad is not None


def test_onnx_export_and_runtime_inference():
    in_channels = 2
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
