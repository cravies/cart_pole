"""
Tests for the two-day policy-gradient briefing.
Written by Claude. Everything in pg.py is yours.

Run:  python -m pytest pg_tests.py -v
      python -m pytest pg_tests.py -v -k "not learns"   (skip the slow training checks)

Interface contract (the only thing the tests assume about pg.py):

    make_policy(obs_dim: int, n_actions: int) -> torch.nn.Module
        module(obs: Tensor[N, obs_dim]) -> logits: Tensor[N, n_actions]

    reward_to_go(rewards: list[float], gamma: float) -> Tensor[T]

    pg_loss(logits: Tensor[N, A], actions: Tensor[N], weights: Tensor[N]) -> Tensor (scalar)

    rloo_advantages(returns: Tensor[K]) -> Tensor[K]              (day 2)
        leave-one-out advantage for each of K episode returns

    make_value_net(obs_dim: int) -> torch.nn.Module            (day 2)
        module(obs: Tensor[N, obs_dim]) -> Tensor[N] or Tensor[N, 1]

    train(cfg: dict) -> list[float]
        per-episode undiscounted returns, one entry per episode
"""
import math
import torch

from pg import make_policy, reward_to_go, pg_loss, train

OBS_DIM, N_ACTIONS = 4, 2


def _cfg(n_episodes=1500, **variance):
    return {
        "env": {"id": "CartPole-v1", "seed": 0},
        "train": {"n_episodes": n_episodes, "episodes_per_update": 10,
                  "gamma": 0.99, "lr": 1e-2},
        "variance": {"reward_to_go": False, "baseline": None,
                     "normalize": False, **variance},
    }


# ---------------------------------------------------------------- returns

def test_reward_to_go_undiscounted():
    g = reward_to_go([1.0, 1.0, 1.0], gamma=1.0)
    assert torch.allclose(g, torch.tensor([3.0, 2.0, 1.0]))


def test_reward_to_go_discounted():
    g = reward_to_go([1.0, 1.0, 1.0], gamma=0.5)
    assert torch.allclose(g, torch.tensor([1.75, 1.5, 1.0]))


def test_reward_to_go_general():
    g = reward_to_go([2.0, -1.0, 5.0], gamma=0.9)
    assert math.isclose(g[2].item(), 5.0)
    assert math.isclose(g[1].item(), -1.0 + 0.9 * 5.0)
    assert math.isclose(g[0].item(), 2.0 + 0.9 * (-1.0 + 0.9 * 5.0))


def test_reward_to_go_shape():
    g = reward_to_go([1.0] * 37, gamma=0.99)
    assert g.shape == (37,)


# ----------------------------------------------------------------- policy

def test_policy_outputs_logits_of_right_shape():
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    logits = pi(torch.randn(16, OBS_DIM))
    assert logits.shape == (16, N_ACTIONS)
    assert torch.isfinite(logits).all()


def test_policy_softmax_is_distribution():
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    probs = torch.softmax(pi(torch.randn(16, OBS_DIM)), dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(16), atol=1e-5)
    assert (probs >= 0).all()


# ------------------------------------------------------------------- loss

def _prob_of(pi, obs, action):
    return torch.softmax(pi(obs), -1)[0, action].item()


def test_positive_weight_raises_action_probability():
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    opt = torch.optim.SGD(pi.parameters(), lr=0.1)
    obs = torch.randn(1, OBS_DIM)
    before = _prob_of(pi, obs, 1)
    for _ in range(20):
        opt.zero_grad()
        pg_loss(pi(obs), torch.tensor([1]), torch.tensor([1.0])).backward()
        opt.step()
    assert _prob_of(pi, obs, 1) > before, "positive weight must push P(action) up"


def test_negative_weight_lowers_action_probability():
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    opt = torch.optim.SGD(pi.parameters(), lr=0.1)
    obs = torch.randn(1, OBS_DIM)
    before = _prob_of(pi, obs, 1)
    for _ in range(20):
        opt.zero_grad()
        pg_loss(pi(obs), torch.tensor([1]), torch.tensor([-1.0])).backward()
        opt.step()
    assert _prob_of(pi, obs, 1) < before, "negative weight must push P(action) down"


def test_loss_is_scalar_and_differentiable():
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    obs = torch.randn(8, OBS_DIM)
    loss = pg_loss(pi(obs), torch.randint(0, N_ACTIONS, (8,)), torch.randn(8))
    assert loss.dim() == 0
    loss.backward()
    assert all(p.grad is not None for p in pi.parameters())


def test_weights_are_treated_as_constants():
    # gradient must not flow into the weights; they are data, not parameters
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    obs = torch.randn(8, OBS_DIM)
    w = torch.randn(8, requires_grad=True)
    pg_loss(pi(obs), torch.randint(0, N_ACTIONS, (8,)), w).backward()
    assert w.grad is None or torch.allclose(w.grad, torch.zeros_like(w))


# --------------------------------------------- the identity behind baselines

def test_score_function_has_zero_expectation():
    # sum_a pi(a|s) grad log pi(a|s) = grad sum_a pi(a|s) = grad 1 = 0
    torch.manual_seed(0)
    pi = make_policy(OBS_DIM, N_ACTIONS)
    obs = torch.randn(1, OBS_DIM)
    logits = pi(obs)
    dist = torch.distributions.Categorical(logits=logits)
    total = None
    for a in range(N_ACTIONS):
        pi.zero_grad(set_to_none=True)
        dist.log_prob(torch.tensor([a])).backward(retain_graph=True)
        g = torch.cat([p.grad.flatten().clone() for p in pi.parameters()])
        term = dist.probs[0, a].detach() * g
        total = term if total is None else total + term
    assert total.abs().max().item() < 1e-5


# ------------------------------------------------------------ day 2: RLOO

def test_rloo_advantage_values():
    # Lambert eq. (48)-(49): b_i = mean of the OTHER returns
    from pg import rloo_advantages
    a = rloo_advantages(torch.tensor([3.0, 1.0, 2.0]))
    assert torch.allclose(a, torch.tensor([1.5, -1.5, 0.0]))


def test_rloo_advantages_sum_to_zero():
    from pg import rloo_advantages
    torch.manual_seed(0)
    a = rloo_advantages(torch.rand(10) * 500)
    assert abs(a.sum().item()) < 1e-3


def test_rloo_equals_scaled_mean_baseline():
    # Lambert eq. (61): RLOO == (K/(K-1)) * (R_i - mean(all R))   [Dr. GRPO]
    from pg import rloo_advantages
    torch.manual_seed(1)
    r = torch.rand(7) * 100
    K = r.numel()
    dr_grpo = (K / (K - 1)) * (r - r.mean())
    assert torch.allclose(rloo_advantages(r), dr_grpo, atol=1e-4)


def test_rloo_two_episodes_is_half_the_difference_doubled():
    # with K=2 each episode's baseline is just the other's return
    from pg import rloo_advantages
    a = rloo_advantages(torch.tensor([10.0, 4.0]))
    assert torch.allclose(a, torch.tensor([6.0, -6.0]))


# ------------------------------------------------------ day 2: value baseline

def test_value_net_fits_a_constant_target():
    from pg import make_value_net
    torch.manual_seed(0)
    v = make_value_net(OBS_DIM)
    opt = torch.optim.Adam(v.parameters(), lr=1e-2)
    obs = torch.randn(64, OBS_DIM)
    target = torch.full((64,), 7.0)
    for _ in range(300):
        opt.zero_grad()
        pred = v(obs).reshape(-1)
        assert pred.shape == (64,), "value net output must flatten to [N]"
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1


# ---------------------------------------------------- learning (slow, ~mins)

def _best_window_mean(xs, k):
    return max(sum(xs[i:i + k]) / k for i in range(len(xs) - k + 1))


def test_learns_cartpole_day1():
    returns = train(_cfg())
    assert len(returns) == 1500
    assert _best_window_mean(returns, 20) >= 195, \
        "vanilla REINFORCE should reach the CartPole-v0 bar (195 over 20 episodes)"


def test_learns_cartpole_day2_rloo():
    # bandit-style: whole-episode return, leave-one-out baseline, no value net
    returns = train(_cfg(n_episodes=3000, reward_to_go=False, baseline="rloo", normalize=True))
    assert _best_window_mean(returns, 100) >= 475, \
        "RLOO + normalization should hit the v1 bar (475 over 100)"


def test_learns_cartpole_day2_value():
    # MDP-style: reward-to-go, learned V(s_t) baseline
    returns = train(_cfg(n_episodes=3000, reward_to_go=True, baseline="value", normalize=True))
    assert _best_window_mean(returns, 100) >= 475, \
        "reward-to-go + value baseline + normalization should hit the v1 bar (475 over 100)"
