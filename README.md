# Computable Infinite Wordle: Formal Model

## Game Setup

Let $b \geq 3$ be the base.  
The secret is a computable real number $S \in [0,1)$ with expansion  
$$
S = 0.s_1 s_2 s_3 \dots, \quad s_i \in \{0,1,\dots,b-1\}.
$$
Let $M_S$ be a Turing machine that computes $s_n$ on input $n$.

---

## One Turn

1. **Player chooses** $k$ distinct positions $p_1 < p_2 < \dots < p_k \in \mathbb{N}$.
2. **Player provides** a Turing machine $M$ such that for each $p_j$, $M(p_j)$ outputs a guess digit $g_{p_j} \in \{0,\dots,b-1\}$.
3. **Feedback** for each $j$:
   - Compute $s_{p_j}$ using $M_S$.
   - Let $d = |g_{p_j} - s_{p_j}|$.
     - If $d = 0$ → 🟩 (exact match)
     - If $d = 1$ → 🟨 (neighbor)
     - If $d \geq 2$ → ⬛ (far)
4. Player receives the $k$-tuple of colors $(c_1,\dots,c_k)$.

---

## Allowed Strategies

- The player may choose $k$ and $M$ adaptively based on all previous feedback.
- The game ends when the player can produce a TM that agrees with $M_S$ on all $n$ (or after finite turns, claims to know the secret and verifies it).

---

## Notes on Feedback Semantics

The "near" condition is **absolute difference 1**, not modular.  
Thus, digit adjacency forms a simple path graph:
$$
0 \text{ — } 1 \text{ — } 2 \text{ — } \dots \text{ — } (b-1).
$$
Digits $0$ and $b-1$ have only one neighbor each; others have two.

---

## Example ($b=4$)

Secret: $S = 0.2301\dots$  
Guess positions $p = 1,2,3,4$, guess digits $1,3,0,1$.

- $|1-2| = 1$ → 🟨
- $|3-3| = 0$ → 🟩
- $|0-0| = 0$ → 🟩
- $|1-1| = 0$ → 🟩

Feedback: 🟨🟩🟩🟩.

---

## Theoretical Questions

1. **Winnability**: Is the game winnable in finite turns for every computable secret?  
   *Potential approach*: Enumerate all TMs, test positions adaptively, eliminate inconsistent ones. Consistency is computably enumerable.

2. **Query Complexity**: Worst‑case number of turns as a function of $b$ and the Kolmogorov complexity of $M_S$?

3. **Base Effect**: Higher $b$ implies more ⬛ feedback (since $|g-s| \geq 2$ is easier to achieve), possibly speeding up convergence.

4. **Alternative Yellow Rule**: If yellow were **modular difference 1** (wrap‑around), adjacency becomes a cycle, changing the information‑theoretic landscape.

---

## Strategy Sketch for Base 4

1. **Initial probe**: Guess all digits as $2$ on positions $1,\dots,k_0$.  
   Feedback reveals exactly which digits are $2$ (🟩), which are $1$ or $3$ (🟨), and which are $0$ (⬛, since $|0-2|=2$).

2. **Disambiguation of 🟨**:  
   For each 🟨 position $p$, guess $1$ there (others fixed).  
   - If feedback turns 🟩 → digit was $1$.  
   - If feedback stays 🟨 → digit was $3$.  
   - If turns ⬛ → digit was $0$ (but that case already ruled out in step 1).

3. **Iterate** with new candidate TM that fits all observed data, test new positions to refute incorrect TMs.

4. **Exploit computability**: The secret’s TM has finite description; candidate TMs can be enumerated and tested.
