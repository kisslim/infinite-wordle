# Computable Infinite Wordle: Two Variance Specifications

## Variance A: Absolute Difference Feedback

**Game Rules:**
- Secret: A computable real number $S = 0.s_1 s_2 s_3 \dots$ in base $b \ge 3$, where $s_i \in \{0,1,\dots,b-1\}$.
- Player's guess: A Turing machine $M$ that outputs digits at queried positions $p_1, p_2, \dots, p_k$.
- Feedback for each guessed digit $g$ at position $p$ compared to secret digit $s$:
  - 🟩 if $g = s$
  - 🟨 if $|g - s| = 1$
  - ⬛ if $|g - s| \ge 2$
- Digit adjacency: Forms a linear graph where endpoints have only one neighbor.
- Example (base 5): Secret digit 0 → yellow for guess 1 only. Secret digit 4 → yellow for guess 3 only.

**Strategy Implications:**
- Endpoint digits (0 and $b-1$) give stronger information when yellow appears.
- As base increases, black feedback becomes more common (>75% of random guesses when $b \ge 4$).
- Distinguishing between distant digit values is efficient.

## Variance B: Modular Difference Feedback

**Game Rules:**
- Same secret and guessing mechanism as Variance A.
- Feedback for each guessed digit $g$ compared to secret digit $s$:
  - 🟩 if $g = s$
  - 🟨 if $(g - s) \mod b \in \{1, b-1\}$ (modular distance 1)
  - ⬛ otherwise
- Digit adjacency: Forms a cyclic graph where every digit has exactly two neighbors.
- Example (base 5): Secret digit 0 → yellow for guesses 1 or 4. Secret digit 2 → yellow for guesses 1 or 3.

**Strategy Implications:**
- Symmetric feedback for all digits creates consistent ambiguity.
- Yellow feedback always presents exactly two possibilities (except when green).
- Higher bases maintain exactly two neighbors per digit, keeping yellow rate constant at approximately 2/$b$ for random guessing.
- More natural generalization from finite Wordle's letter-position constraints.

---

## Comparative Analysis

| Aspect | Variance A (Absolute) | Variance B (Modular) |
|--------|----------------------|----------------------|
| **Adjacency Graph** | Linear path graph | Cyclic graph |
| **Endpoint Behavior** | Digits 0 and $b-1$ have one neighbor | All digits have two neighbors |
| **Feedback Ambiguity** | Lower for endpoints, higher for middle digits | Uniform across all digits |
| **Yellow Probability** | $\frac{2}{b}$ for middle digits, $\frac{1}{b}$ for endpoints (average $< \frac{2}{b}$) | Exactly $\frac{2}{b}$ for all digits |
| **Strategy Complexity** | Asymmetric cases require special handling | Symmetric strategy works uniformly |
| **Base Dependence** | Black feedback increases faster with $b$ | Yellow feedback scales as $2/b$ regardless of $b$ |

---

## Example Gameplay in Base 4

**Secret:** $0.2301\dots$

**Variance A (Absolute Difference):**
```
Turn 1: Guess positions 1-4, digits "1310"
Position 1: guess 1, secret 2 → |1-2|=1 → 🟨
Position 2: guess 3, secret 3 → |3-3|=0 → 🟩
Position 3: guess 1, secret 0 → |1-0|=1 → 🟨
Position 4: guess 0, secret 1 → |0-1|=1 → 🟨
Feedback: 🟨🟩🟨🟨
```

**Variance B (Modular Difference):**
```
Turn 1: Guess positions 1-4, digits "1310"
Position 1: guess 1, secret 2 → (1-2) mod 4 = 3 → 🟨 (since 3 ≡ b-1)
Position 2: guess 3, secret 3 → (3-3) mod 4 = 0 → 🟩
Position 3: guess 1, secret 0 → (1-0) mod 4 = 1 → 🟨
Position 4: guess 0, secret 1 → (0-1) mod 4 = 3 → 🟨
Feedback: 🟨🟩🟨🟨 (same in this case, but interpretation differs)
```

**Interpretation Difference:**
- Variance A: Position 1 yellow means secret digit is exactly 2 (only neighbor of 1)
- Variance B: Position 1 yellow means secret digit is 0 or 2 (both modular neighbors of 1)

---

## Strategic Considerations

### For Variance A:
1. **Endpoint Probing:** Early guesses should test digits near 0 and $b-1$ to exploit their informational advantage.
2. **Binary Search Analogy:** With yellow feedback, you can often reduce possibilities by half.
3. **Optimal First Guess:** In base 10, guessing digit 5 yields maximum information (neighbors 4 and 6).

### For Variance B:
1. **Uniform Strategy:** Same approach works for all digit values.
2. **Cyclic Elimination:** Yellow feedback always gives exactly two possibilities to distinguish.
3. **Optimal First Guess:** All digits are informationally equivalent.

---

## Theoretical Questions for Both Variances

1. **Winnability:** Is every computable secret determinable in finite time?
   - Both variances: Yes, via enumeration of TMs with consistency checking.
   
2. **Minimal Turns:** What's the worst-case number of turns needed?
   - Variance A: Potentially fewer turns due to endpoint information.
   - Variance B: More consistent but possibly more turns due to uniform ambiguity.

3. **Base Scaling:** How does difficulty scale with base $b$?
   - Variance A: Gets easier as $b$ increases (more black feedback).
   - Variance B: Difficulty scales approximately linearly with $b$.

4. **Optimal Algorithms:** Is there a strategy minimizing worst-case turns?
   - Both are open problems, though halving strategies seem natural.

---

## Implementation Sketch (Pseudocode)

```python
class InfiniteWordle:
    def __init__(self, secret_tm, base, variance):
        self.secret_tm = secret_tm
        self.base = base
        self.variance = variance  # 'absolute' or 'modular'
        self.tested_positions = {}
    
    def get_feedback(self, positions, guess_digits):
        feedback = []
        for pos, guess in zip(positions, guess_digits):
            secret = self.secret_tm.compute_digit(pos)
            if guess == secret:
                feedback.append('🟩')
            elif self._is_near(guess, secret):
                feedback.append('🟨')
            else:
                feedback.append('⬛')
        return feedback
    
    def _is_near(self, g, s):
        if self.variance == 'absolute':
            return abs(g - s) == 1
        else:  # modular
            diff = (g - s) % self.base
            return diff == 1 or diff == self.base - 1
```

**Research Directions:**
1. Complexity of optimal strategies for each variance
2. Average-case analysis for random computable secrets
3. Effect of restricting TM complexity classes
4. Relationship to oracle-based learning algorithms
