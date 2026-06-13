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


def test_fisher_probe_token_count_override_rescales():
    """The token_count_override rescales the probe by sqrt(real/override).

    Guards the micro-batched AURA path: each micro-batch must normalize by
    the GLOBAL token count, or the gradient summed across M micro-batches is
    sqrt(M)-inflated (and the squared cost M-inflated). Deterministic, bit-
    level — same seed gives the same Rademacher draw.
    """
    import math
    import torch
    from prismaquant.kl_fisher import fisher_probe_scalar

    torch.manual_seed(0)
    logits = torch.randn(4, 8, 16)  # B=4, T=8, V=16; scope="all" -> N=32
    kw = dict(seed=5, token_scope="all", distribution="rademacher")
    base = fisher_probe_scalar(logits, **kw)
    # override == real count is an exact no-op
    same = fisher_probe_scalar(logits, token_count_override=32, **kw)
    assert torch.equal(same, base)
    # override == 4x real count scales the probe (hence the scalar) by 1/2
    quad = fisher_probe_scalar(logits, token_count_override=4 * 32, **kw)
    assert torch.allclose(quad, base * 0.5, rtol=1e-5, atol=0.0)
