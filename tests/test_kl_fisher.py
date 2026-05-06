import torch
import torch.nn.functional as F

from prismaquant.kl_fisher import fisher_probe_scalar, fisher_quadratic_form


def _forward_kl(teacher_logits, student_logits, *, temperature=1.0, token_scope="last"):
    if token_scope == "last":
        teacher_logits = teacher_logits[..., -1:, :]
        student_logits = student_logits[..., -1:, :]
    elif token_scope == "causal":
        teacher_logits = teacher_logits[..., :-1, :]
        student_logits = student_logits[..., :-1, :]
    elif token_scope != "all":
        raise ValueError(token_scope)
    teacher_log_probs = F.log_softmax(teacher_logits.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()


def test_fisher_quadratic_matches_forward_kl_second_order():
    torch.manual_seed(11)
    logits = torch.randn(2, 4, 7)
    delta = 0.05 * torch.randn(2, 4, 7)

    actual = _forward_kl(
        logits,
        logits + delta,
        temperature=1.7,
        token_scope="all",
    )
    approx = fisher_quadratic_form(
        logits,
        delta,
        temperature=1.7,
        token_scope="all",
    )

    assert actual.item() > 0.0
    assert approx.item() > 0.0
    torch.testing.assert_close(
        actual,
        approx,
        rtol=0.08,
        atol=2e-5,
    )


def test_fisher_probe_gradient_is_centered_and_respects_last_scope():
    torch.manual_seed(13)
    logits = torch.randn(1, 3, 11, requires_grad=True)

    scalar = fisher_probe_scalar(
        logits,
        seed=5,
        token_scope="last",
        temperature=1.3,
        distribution="rademacher",
    )
    scalar.backward()

    grad = logits.grad
    assert grad is not None
    assert torch.count_nonzero(grad[:, :-1, :]).item() == 0
    assert torch.count_nonzero(grad[:, -1:, :]).item() > 0
    torch.testing.assert_close(
        grad.sum(dim=-1),
        torch.zeros_like(grad.sum(dim=-1)),
        atol=1e-6,
        rtol=1e-6,
    )
