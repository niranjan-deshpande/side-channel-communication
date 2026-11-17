"""Analytic baseline for untrusted monitoring: a colluding monitor with no information about which attacks are real.

Protocol: the deployment contains A real attacks and H honeypots in random order (benign actions are irrelevant to the
outcome here). The monitor sees each attack-looking item and, being uninformed, passes it with probability q and flags
it otherwise. A flagged real attack is audited and caught: blue wins. A passed honeypot is noticed: blue wins. A passed
real attack before either of those: red wins.

Red wins iff the first real attack is passed and every honeypot before it was flagged. With k honeypots ahead of the
first real attack, that probability is q (1 - q)^k, and k follows the distribution of honeypots preceding the first of
A attacks in a uniformly random order.

usage: python baseline.py [--attacks A] [--honeypots H...]
"""
import argparse
from math import comb


def p_red_wins(q, A, H):
    total = 0.0
    for k in range(H + 1):
        # P(exactly k honeypots before the first attack) in a random order of A attacks and H honeypots
        pk = comb(H + A - k - 1, A - 1) / comb(H + A, A)
        total += pk * q * (1 - q) ** k
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", type=int, default=5)
    ap.add_argument("--honeypots", type=int, nargs="+", default=[0, 5, 20, 50])
    args = ap.parse_args()
    qs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    print(f"P(red wins), A = {args.attacks} real attacks, uninformed colluder passing each item with prob q")
    print("H \\ q   " + "".join(f"{q:>7.2f}" for q in qs))
    for H in args.honeypots:
        print(f"{H:<8d}" + "".join(f"{p_red_wins(q, args.attacks, H):7.3f}" for q in qs))
    for H in args.honeypots:
        best = max((p_red_wins(q / 1000, args.attacks, H), q / 1000) for q in range(1001))
        print(f"H={H}: best uninformed colluder wins with p = {best[0]:.3f} (q = {best[1]:.3f})")
    print("An informed colluder (passes real attacks, flags honeypots) wins with p = 1 in this model.")


if __name__ == "__main__":
    main()
