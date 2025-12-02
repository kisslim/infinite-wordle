# computable infinite Wordle
a game to guess numbers between 0 and 1.

- The secret is a **computable real number** in \([0,1)\).
- Base \( b \ge 3 \).
- You guess by **providing a Turing machine** that, given \( k \) integers \(n_1, n_2, \dots, n_k\), outputs \( k \) digits of your guessed number at those positions.
- After each guess, you get feedback for each digit: 🟩 (exact match), 🟨 (digit differs by 1 modulo base? or absolute difference = 1?), ⬛ (absolute difference ≥ 2).
- Goal: deduce the secret’s digits in all positions (or up to any desired precision).

---

## **Formalizing the game**

Let’s set \( b \ge 3 \), secret \( S = 0.s_1 s_2 s_3\dots \) (each \( s_i \in \{0,1,\dots,b-1\} \)), \( S \) computable.

**One turn:**
1. Player chooses \( k \) positions \( p_1 < p_2 < \dots < p_k \) (could be any positive integers).
2. Player provides a Turing machine \( M \) that on input \( p_j \) outputs guess digit \( g_{p_j} \in \{0,\dots,b-1\} \).
3. For each \( j \):
   - Compute \( s_{p_j} \) (possible since \( S \) is computable).
   - Compare \( g_{p_j} \) with \( s_{p_j} \):
     - If \( g_{p_j} = s_{p_j} \) → 🟩
     - If \( |g_{p_j} - s_{p_j}| = 1 \) → 🟨
     - Else (difference ≥ 2) → ⬛
4. Player receives tuple of colored feedback for \( p_1, \dots, p_k \).

**Rules:**
- Player can adaptively choose \( k \) and \( M \) each turn based on previous feedback.
- Game ends when player can produce a Turing machine that exactly outputs \( s_n \) for all \( n \) (or when they claim to know the secret and verify).

---

## **Why computable secret?**
If secret is computable, then checking equality with guesses is computable (though in reality, checking if two TMs compute the same function is not computable in general — but here the secret TM is fixed, so feedback for chosen positions is computable).

The “feedback rule” avoids giving exact digit when wrong, giving only near/far info.

---

## **Strategic depth**
In base 3 with the difference rule:
- Digit value difference = 0 → 🟩  
- Digit value difference = 1 → 🟨  
- Digit value difference ≥ 2 → ⬛

For base \( b > 3 \), the “near” (🟨) could be \( \pm 1 \) mod \( b \) or absolute difference = 1; but if it’s mod \( b \), then every digit has exactly 2 neighbors (e.g., in base 10, digit 0’s neighbors are 1 and 9 if wrap-around).  
You said “near” means ±1, not modulo wrap, so for large bases, digits 0 and b-1 have only one neighbor, others have two.

---

## **Example play in base 4**
Secret: \( 0.2301\dots \)  
Guess positions \( p = 1,2,3,4 \), guess digits \( 1,3,0,1 \):

Pos1: guess 1, secret 2 → diff 1 → 🟨  
Pos2: guess 3, secret 3 → diff 0 → 🟩  
Pos3: guess 0, secret 0 → 🟩  
Pos4: guess 1, secret 1 → 🟩

Feedback: 🟨🟩🟩🟩

---

## **Key challenge for the player**
The Turing machine \( M \) encodes a **function** \( f: \mathbb{N} \to \{0,\dots,b-1\} \) (restricted to guessed positions).  
So the player can guess a **computable pattern** for the secret’s digits, test a few positions, refine.

Because the secret is computable, there exists a TM for it.  
So game is: **find that TM** using limited “colored digit” queries.

---

## **Interesting theoretical questions**
1. **Is this game winnable in finite steps for any computable secret?**  
   Possibly yes — you can enumerate all TMs and test positions until only one is consistent. But “consistent” here means: for all tested positions, |g−s| yields correct color. That’s a c.e. condition.

2. **Lower bound on needed turns** — in worst case, could take many turns because yellow feedback is ambiguous.

3. **Effect of base** — higher base makes ⬛ more common (since difference ≥ 2 is easier), maybe easier to zero in?

4. **Can player ask for arbitrary positions each turn?** Yes — adaptive.

---
