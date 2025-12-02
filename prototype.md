# Computable Infinite Wordle Project

## Project Structure

```
infinite-wordle/
├── infinite_wordle/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── game.py          # Main game logic
│   │   ├── feedback.py      # Feedback generation
│   │   ├── secret.py        # Secret number generation
│   │   └── turing.py        # Turing machine simulator
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── sandbox.py       # PyPy sandboxing
│   │   ├── formula_parser.py # SymPy formula parsing
│   │   └── validation.py    # Input validation
│   └── cli/
│       ├── __init__.py
│       ├── main.py          # CLI entry point
│       ├── commands.py      # CLI commands
│       └── display.py       # Game display
├── tests/
│   ├── __init__.py
│   ├── test_game.py
│   ├── test_feedback.py
│   └── test_integration.py
├── examples/
│   ├── formula_secrets.py
│   ├── custom_tm_secrets.py
│   └── predefined_games.py
├── pyproject.toml
├── README.md
├── .env.example
└── requirements.txt
```

## Implementation

### 1. Core Game Files

**`infinite_wordle/core/game.py`**:
```python
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import sympy as sp
from .feedback import FeedbackSystem
from .secret import SecretGenerator
from .turing import TuringMachine

class Variance(Enum):
    ABSOLUTE = "absolute"
    MODULAR = "modular"

@dataclass
class TurnResult:
    positions: List[int]
    guesses: List[int]
    feedback: List[str]
    secret_digits: List[int]

class InfiniteWordleGame:
    def __init__(
        self,
        base: int = 10,
        variance: Variance = Variance.ABSOLUTE,
        secret_source: Any = None,
        max_positions: int = 1000,
        max_turns: int = 100
    ):
        if base < 3:
            raise ValueError("Base must be at least 3")
        
        self.base = base
        self.variance = variance
        self.max_positions = max_positions
        self.max_turns = max_turns
        self.turn_history: List[TurnResult] = []
        
        # Initialize secret
        if secret_source is None:
            self.secret_generator = SecretGenerator(base)
            self.secret_source = self.secret_generator.generate_random()
        elif isinstance(secret_source, (str, sp.Expr)):
            self.secret_source = SecretGenerator.create_from_formula(secret_source, base)
        elif isinstance(secret_source, dict):
            self.secret_source = TuringMachine.from_dict(secret_source)
        else:
            self.secret_source = secret_source
        
        self.feedback_system = FeedbackSystem(base, variance)
    
    def guess_positions(self, positions: List[int], guess_digits: List[int]) -> TurnResult:
        """Make a guess for specific positions"""
        if len(positions) != len(guess_digits):
            raise ValueError("Positions and guesses must have same length")
        
        # Get secret digits
        secret_digits = []
        for pos in positions:
            if 1 <= pos <= self.max_positions:
                secret_digits.append(self._get_secret_digit(pos))
            else:
                raise ValueError(f"Position {pos} out of range [1, {self.max_positions}]")
        
        # Get feedback
        feedback = []
        for guess, secret in zip(guess_digits, secret_digits):
            feedback.append(self.feedback_system.get_feedback(guess, secret))
        
        result = TurnResult(
            positions=positions.copy(),
            guesses=guess_digits.copy(),
            feedback=feedback.copy(),
            secret_digits=secret_digits.copy()
        )
        
        self.turn_history.append(result)
        return result
    
    def _get_secret_digit(self, position: int) -> int:
        """Get digit at position from secret source"""
        if hasattr(self.secret_source, 'get_digit'):
            return self.secret_source.get_digit(position)
        elif hasattr(self.secret_source, '__call__'):
            return self.secret_source(position)
        else:
            # Assume it's a SymPy expression or formula
            from ..utils.formula_parser import evaluate_formula
            return evaluate_formula(self.secret_source, position, self.base)
    
    def is_solved(self, positions: List[int] = None) -> bool:
        """Check if all specified positions are solved"""
        if not self.turn_history:
            return False
        
        if positions is None:
            # Check all positions that have been guessed
            positions = set()
            for turn in self.turn_history:
                positions.update(turn.positions)
            positions = list(positions)
        
        # Build knowledge base
        knowledge = {}
        for turn in self.turn_history:
            for pos, guess, fb, secret in zip(
                turn.positions, turn.guesses, turn.feedback, turn.secret_digits
            ):
                if pos in positions:
                    if fb == '🟩':
                        knowledge[pos] = secret
                    elif pos not in knowledge:
                        # For yellow/black, we need to track possibilities
                        pass
        
        # Check if all positions are known
        return all(pos in knowledge for pos in positions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get game statistics"""
        if not self.turn_history:
            return {}
        
        total_guesses = sum(len(turn.guesses) for turn in self.turn_history)
        greens = sum(fb.count('🟩') for turn in self.turn_history)
        yellows = sum(fb.count('🟨') for turn in self.turn_history)
        blacks = sum(fb.count('⬛') for turn in self.turn_history)
        
        return {
            'turns': len(self.turn_history),
            'total_guesses': total_guesses,
            'greens': greens,
            'yellows': yellows,
            'blacks': blacks,
            'accuracy': greens / total_guesses if total_guesses > 0 else 0,
            'solved': self.is_solved()
        }
```

**`infinite_wordle/core/feedback.py`**:
```python
class FeedbackSystem:
    def __init__(self, base: int, variance: str):
        self.base = base
        self.variance = variance
    
    def get_feedback(self, guess: int, secret: int) -> str:
        """Get feedback emoji for a guess"""
        if guess == secret:
            return '🟩'
        elif self._is_near(guess, secret):
            return '🟨'
        else:
            return '⬛'
    
    def _is_near(self, guess: int, secret: int) -> bool:
        """Check if guess is near secret based on variance"""
        if self.variance == 'absolute':
            return abs(guess - secret) == 1
        else:  # modular
            diff = (guess - secret) % self.base
            return diff == 1 or diff == self.base - 1
    
    def get_possible_secrets(self, guess: int, feedback: str) -> List[int]:
        """Get all possible secret digits given guess and feedback"""
        if feedback == '🟩':
            return [guess]
        elif feedback == '🟨':
            if self.variance == 'absolute':
                # Linear adjacency
                neighbors = []
                if guess > 0:
                    neighbors.append(guess - 1)
                if guess < self.base - 1:
                    neighbors.append(guess + 1)
                return neighbors
            else:
                # Cyclic adjacency
                return [(guess - 1) % self.base, (guess + 1) % self.base]
        else:  # ⬛
            all_digits = set(range(self.base))
            all_digits.discard(guess)
            # Remove neighbors for yellow
            if self.variance == 'absolute':
                if guess > 0:
                    all_digits.discard(guess - 1)
                if guess < self.base - 1:
                    all_digits.discard(guess + 1)
            else:
                all_digits.discard((guess - 1) % self.base)
                all_digits.discard((guess + 1) % self.base)
            return list(all_digits)
```

**`infinite_wordle/core/secret.py`**:
```python
import sympy as sp
import random
from typing import Union, Callable
import math

class SecretGenerator:
    def __init__(self, base: int = 10):
        self.base = base
    
    @staticmethod
    def create_from_formula(formula: Union[str, sp.Expr], base: int) -> Callable[[int], int]:
        """Create a secret source from SymPy formula"""
        if isinstance(formula, str):
            formula = sp.sympify(formula)
        
        def get_digit(n: int) -> int:
            # Formula should produce a number in [0, 1)
            x = sp.N(formula.subs('n', n))
            if isinstance(x, sp.Float):
                value = float(x)
            else:
                value = float(sp.N(x))
            
            # Get nth digit in specified base
            value = abs(value) % 1  # Fractional part
            for _ in range(n):
                value = (value * base) % 1
            digit = int(value * base)
            return digit % base
        
        return get_digit
    
    def generate_random(self, seed: int = None) -> Callable[[int], int]:
        """Generate random computable sequence"""
        if seed is not None:
            random.seed(seed)
        
        # Generate a random algebraic formula or simple pattern
        patterns = [
            lambda n: (n * 13) % self.base,
            lambda n: (n ** 2) % self.base,
            lambda n: ((n * 7) + 3) % self.base,
            lambda n: int(math.sin(n) * 1000) % self.base,
        ]
        
        return random.choice(patterns)
    
    @staticmethod
    def from_champernowne(base: int) -> Callable[[int], int]:
        """Champernowne constant in given base"""
        def get_digit(n: int) -> int:
            # Find which block the digit is in
            length = 1
            start = 0
            count = base - 1  # Numbers starting with non-zero
            
            while n >= count * length:
                n -= count * length
                length += 1
                count *= base
            
            # Find the actual number
            num = base ** (length - 1) + n // length
            digit_pos = n % length
            
            # Get the specific digit
            for _ in range(length - digit_pos - 1):
                num //= base
            return num % base
        
        return get_digit
```

### 2. Sandbox and Security

**`infinite_wordle/utils/sandbox.py`**:
```python
import tempfile
import os
import subprocess
import sys
from typing import Optional, Any
import ast

class PyPySandbox:
    """Sandbox for executing user code safely using PyPy"""
    
    def __init__(self, timeout: int = 5, memory_limit: int = 256):
        self.timeout = timeout
        self.memory_limit = memory_limit  # MB
    
    def execute_turing_machine(self, code: str, input_data: Any) -> Optional[Any]:
        """Execute Turing machine code in sandbox"""
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap code in sandbox
            sandboxed_code = f"""
import sys
import resource

# Set memory limit
resource.setrlimit(resource.RLIMIT_AS, 
                  ({self.memory_limit * 1024 * 1024}, 
                   {self.memory_limit * 1024 * 1024}))

# Disable dangerous modules
for mod in ['os', 'sys', 'subprocess', 'shutil', 'socket']:
    sys.modules[mod] = None

# User code
{code}

# Execution
result = compute_digit({input_data})
print(result)
"""
            f.write(sandboxed_code)
            temp_file = f.name
        
        try:
            # Execute with PyPy
            result = subprocess.run(
                ['pypy3', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                return int(result.stdout.strip())
            else:
                print(f"Execution error: {result.stderr}", file=sys.stderr)
                return None
                
        except subprocess.TimeoutExpired:
            print("Execution timed out", file=sys.stderr)
            return None
        finally:
            os.unlink(temp_file)
    
    def validate_code(self, code: str) -> bool:
        """Validate that code is safe (basic check)"""
        try:
            tree = ast.parse(code)
            
            # Check for dangerous constructs
            for node in ast.walk(tree):
                # No imports
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    return False
                # No function definitions except allowed ones
                if isinstance(node, ast.FunctionDef):
                    if node.name not in ['compute_digit', '__init__']:
                        return False
                # No classes
                if isinstance(node, ast.ClassDef):
                    return False
            
            # Check that compute_digit function exists
            has_compute_digit = any(
                isinstance(node, ast.FunctionDef) and node.name == 'compute_digit'
                for node in tree.body
            )
            
            return has_compute_digit
            
        except SyntaxError:
            return False
```

### 3. Formula Parser

**`infinite_wordle/utils/formula_parser.py`**:
```python
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from typing import Union, Dict, Any

class FormulaParser:
    """Parse and evaluate mathematical formulas"""
    
    def __init__(self, base: int = 10):
        self.base = base
        self.symbols = {'n': sp.Symbol('n'), 'pi': sp.pi, 'e': sp.E}
    
    def parse(self, formula: str) -> sp.Expr:
        """Parse formula string to SymPy expression"""
        try:
            # Allow common mathematical functions
            allowed_functions = [
                'sin', 'cos', 'tan', 'log', 'exp', 'sqrt', 'abs',
                'floor', 'ceil', 'mod'
            ]
            
            return parse_expr(
                formula,
                local_dict={**self.symbols, **{f: getattr(sp, f) for f in allowed_functions}}
            )
        except Exception as e:
            raise ValueError(f"Invalid formula: {e}")
    
    def evaluate_at_position(self, formula: Union[str, sp.Expr], n: int) -> int:
        """Evaluate formula at position n and extract digit"""
        if isinstance(formula, str):
            expr = self.parse(formula)
        else:
            expr = formula
        
        # Evaluate expression
        value = sp.N(expr.subs('n', n))
        
        # Get nth digit in specified base
        value = float(abs(value) % 1)  # Fractional part
        
        for _ in range(n):
            value = (value * self.base) % 1
        
        digit = int(value * self.base)
        return digit % self.base
    
    @staticmethod
    def create_formula_function(formula: str, base: int) -> callable:
        """Create a function that computes digits from formula"""
        parser = FormulaParser(base)
        expr = parser.parse(formula)
        
        def get_digit(n: int) -> int:
            return parser.evaluate_at_position(expr, n)
        
        return get_digit
```

### 4. CLI Interface

**`infinite_wordle/cli/main.py`**:
```python
import click
import os
from typing import Optional
from .commands import (
    new_game,
    make_guess,
    show_history,
    show_stats,
    solve_auto,
    analyze_position
)

@click.group()
@click.option('--base', type=int, default=None,
              help='Number base (default: from ENV or 10)')
@click.option('--variance', type=click.Choice(['absolute', 'modular']),
              default=None, help='Feedback variance')
@click.option('--max-positions', type=int, default=None,
              help='Maximum positions to allow')
@click.pass_context
def cli(ctx, base, variance, max_positions):
    """Computable Infinite Wordle - Guess digits of a computable real number"""
    # Load from environment variables if not provided
    base = base or int(os.getenv('WORDLE_BASE', '10'))
    variance = variance or os.getenv('WORDLE_VARIANCE', 'absolute')
    max_positions = max_positions or int(os.getenv('WORDLE_MAX_POSITIONS', '1000'))
    
    ctx.ensure_object(dict)
    ctx.obj['base'] = base
    ctx.obj['variance'] = variance
    ctx.obj['max_positions'] = max_positions

@cli.command()
@click.option('--secret', type=str, default=None,
              help='Secret formula or TM description')
@click.option('--random', is_flag=True, help='Use random secret')
@click.option('--champernowne', is_flag=True, help='Use Champernowne constant')
@click.option('--from-file', type=click.Path(exists=True),
              help='Load secret from JSON file')
@click.pass_context
def new(ctx, secret, random, champernowne, from_file):
    """Start a new game"""
    new_game(ctx, secret, random, champernowne, from_file)

@cli.command()
@click.argument('positions', type=int, nargs=-1, required=True)
@click.argument('digits', type=int, nargs=-1, required=True)
@click.pass_context
def guess(ctx, positions, digits):
    """Make a guess for specific positions"""
    make_guess(ctx, positions, digits)

@cli.command()
@click.pass_context
def history(ctx):
    """Show game history"""
    show_history(ctx)

@cli.command()
@click.pass_context
def stats(ctx):
    """Show game statistics"""
    show_stats(ctx)

@cli.command()
@click.option('--positions', type=int, multiple=True,
              help='Positions to solve (default: all guessed)')
@click.option('--strategy', type=click.Choice(['binary', 'linear', 'optimal']),
              default='binary', help='Solving strategy')
@click.pass_context
def solve(ctx, positions, strategy):
    """Attempt to solve automatically"""
    solve_auto(ctx, positions, strategy)

@cli.command()
@click.argument('position', type=int)
@click.pass_context
def analyze(ctx, position):
    """Analyze possibilities for a position"""
    analyze_position(ctx, position)

if __name__ == '__main__':
    cli(obj={})
```

**`infinite_wordle/cli/display.py`**:
```python
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def display_turn(positions: List[int], guesses: List[int], feedback: List[str],
                 secret_digits: List[int] = None, show_secret: bool = False):
    """Display a turn result beautifully"""
    table = Table(title=f"Turn Result - Positions {positions}")
    table.add_column("Position", style="cyan")
    table.add_column("Guess", style="magenta")
    table.add_column("Feedback", style="bold")
    if show_secret and secret_digits:
        table.add_column("Secret", style="green")
    
    for i, (pos, guess, fb) in enumerate(zip(positions, guesses, feedback)):
        row = [str(pos), str(guess), fb]
        if show_secret and secret_digits:
            row.append(str(secret_digits[i]))
        table.add_row(*row)
    
    console.print(table)

def display_knowledge(knowledge: dict, base: int):
    """Display current knowledge state"""
    from rich.tree import Tree
    
    tree = Tree("📊 Current Knowledge")
    
    for pos, info in knowledge.items():
        if isinstance(info, int):  # Known digit
            branch = tree.add(f"[green]Position {pos}: {info} 🟩[/green]")
        elif isinstance(info, list):  # Possible digits
            branch = tree.add(f"[yellow]Position {pos}: {info} 🟨[/yellow]")
        else:  # Unknown
            branch = tree.add(f"[grey]Position {pos}: 0-{base-1} ⬛[/grey]")
    
    console.print(Panel(tree, title="Knowledge Base"))

def display_help():
    """Display help information"""
    help_text = """
[b]Computable Infinite Wordle[/b]

[b]Game Rules:[/b]
- Secret is a computable real number in base b ≥ 3
- Guess digits at specific positions
- Feedback:
  🟩 = correct digit
  🟨 = adjacent digit (distance 1)
  ⬛ = distant digit

[b]Variances:[/b]
1. [i]Absolute[/i]: Linear adjacency (0-1-2-...-(b-1))
2. [i]Modular[/i]: Cyclic adjacency (0-1-2-...-(b-1)-0)

[b]Commands:[/b]
  new       - Start new game
  guess     - Make guess
  history   - Show history
  stats     - Show statistics
  solve     - Auto-solve
  analyze   - Analyze position
    """
    
    console.print(Panel(help_text, title="Help"))
```

### 5. Example Usage Files

**`examples/formula_secrets.py`**:
```python
#!/usr/bin/env python3
"""Example secret formulas for Infinite Wordle"""

from infinite_wordle.utils.formula_parser import FormulaParser

# Example formulas that produce computable sequences
FORMULAS = {
    # Simple patterns
    "simple_linear": "n % base",
    "quadratic": "(n**2) % base",
    "alternating": "(-1)**n / 2 + 0.5",  # Produces 0,1,0,1,...
    
    # Mathematical constants
    "pi_digits": "pi * (n+1)",  # Digits of pi
    "e_digits": "e * n",
    "golden_ratio": "(1 + sqrt(5))/2 * n",
    
    # Trigonometric patterns
    "sine_wave": "sin(n)",
    "cosine_mod": "cos(pi * n / 7)",
    
    # More complex
    "fibonacci_mod": "fibonacci(n) % base",
    "prime_gaps": "prime(n+1) - prime(n)",
}

def create_secret_from_formula(formula_name: str, base: int = 10):
    """Create a secret source from named formula"""
    if formula_name not in FORMULAS:
        raise ValueError(f"Unknown formula: {formula_name}")
    
    parser = FormulaParser(base)
    formula = FORMULAS[formula_name]
    
    # Replace 'base' with actual base value
    if 'base' in formula:
        formula = formula.replace('base', str(base))
    
    return parser.create_formula_function(formula, base)

if __name__ == "__main__":
    # Test each formula
    base = 10
    for name in FORMULAS:
        try:
            secret_func = create_secret_from_formula(name, base)
            digits = [secret_func(i+1) for i in range(5)]
            print(f"{name:20} {digits}")
        except:
            print(f"{name:20} [ERROR]")
```

**`examples/predefined_games.py`**:
```python
#!/usr/bin/env python3
"""Predefined game configurations"""

from infinite_wordle.core.game import InfiniteWordleGame, Variance
from infinite_wordle.core.secret import SecretGenerator

PREDEFINED_GAMES = {
    "beginner_abs": {
        "base": 4,
        "variance": Variance.ABSOLUTE,
        "secret": "n % 4",
        "description": "Simple modulo pattern (base 4, absolute)"
    },
    "beginner_mod": {
        "base": 4,
        "variance": Variance.MODULAR,
        "secret": "(n**2) % 4",
        "description": "Quadratic pattern (base 4, modular)"
    },
    "intermediate_pi": {
        "base": 10,
        "variance": Variance.ABSOLUTE,
        "secret": "pi * n",
        "description": "Digits related to pi (base 10)"
    },
    "challenge_champernowne": {
        "base": 10,
        "variance": Variance.MODULAR,
        "secret": "champernowne",
        "description": "Champernowne constant (challenging)"
    },
    "high_base": {
        "base": 16,
        "variance": Variance.ABSOLUTE,
        "secret": "sin(n) * 1000",
        "description": "Hexadecimal challenge"
    }
}

def create_predefined_game(name: str) -> InfiniteWordleGame:
    """Create a predefined game"""
    if name not in PREDEFINED_GAMES:
        raise ValueError(f"Unknown game: {name}")
    
    config = PREDEFINED_GAMES[name]
    
    # Handle special secret types
    if config["secret"] == "champernowne":
        secret_source = SecretGenerator.from_champernowne(config["base"])
    else:
        secret_source = config["secret"]
    
    return InfiniteWordleGame(
        base=config["base"],
        variance=config["variance"],
        secret_source=secret_source
    )

def list_games():
    """List all predefined games"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    table = Table(title="Predefined Games")
    table.add_column("Name", style="cyan")
    table.add_column("Base", style="magenta")
    table.add_column("Variance", style="yellow")
    table.add_column("Description")
    
    for name, config in PREDEFINED_GAMES.items():
        table.add_row(
            name,
            str(config["base"]),
            config["variance"].value,
            config["description"]
        )
    
    console.print(table)

if __name__ == "__main__":
    list_games()
```

### 6. Configuration and Setup

**`pyproject.toml`**:
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "infinite-wordle"
version = "0.1.0"
description = "Computable Infinite Wordle Game"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}

authors = [
    {name = "Your Name", email = "your.email@example.com"}
]

dependencies = [
    "click>=8.0.0",
    "rich>=10.0.0",
    "sympy>=1.9",
    "python-dotenv>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "mypy>=0.910",
]

[project.scripts]
infinite-wordle = "infinite_wordle.cli.main:cli"

[tool.black]
line-length = 88
target-version = ['py38']
```

**`README.md`**:
```markdown
# Computable Infinite Wordle

A mathematical guessing game where you try to determine the digits of a computable real number with Wordle-style feedback.

## Features

- **Two variance systems**: Absolute difference (linear) and Modular difference (cyclic)
- **Multiple secret sources**: Mathematical formulas, Turing machines, or random sequences
- **Safe execution**: PyPy sandboxing for user-provided Turing machines
- **Rich CLI interface**: Beautiful terminal display with `rich`
- **Mathematical foundation**: Built on SymPy for formula evaluation

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/infinite-wordle.git
cd infinite-wordle
pip install -e .

# Or install directly
pip install infinite-wordle
```

## Quick Start

```bash
# Start a new game (using environment variables)
export WORDLE_BASE=10
export WORDLE_VARIANCE=absolute
infinite-wordle new --random

# Or with options
infinite-wordle --base 8 --variance modular new --champernowne

# Make a guess
infinite-wordle guess 1 2 3 4 5  # Positions
               5 2 7 1 8        # Guesses

# Show statistics
infinite-wordle stats

# Auto-solve
infinite-wordle solve --strategy binary
```

## Environment Variables

- `WORDLE_BASE`: Number base (default: 10)
- `WORDLE_VARIANCE`: `absolute` or `modular` (default: `absolute`)
- `WORDLE_MAX_POSITIONS`: Maximum position allowed (default: 1000)

## Examples

See the `examples/` directory for:
- Formula-based secrets
- Predefined game configurations
- Custom Turing machine implementations

## Game Theory

This implementation explores:
1. Computability theory applied to games
2. Information theory of Wordle-style feedback
3. Strategy optimization for different variances
4. Complexity of determining computable sequences

## License

MIT
```

## Usage Examples

### 1. Starting a Game
```bash
# Using formula
infinite-wordle new --secret "sin(n) * 1000"

# Random secret
infinite-wordle new --random

# Predefined game
python -m examples.predefined_games
```

### 2. Making Guesses
```bash
# Guess digits at positions 1, 3, 5
infinite-wordle guess 1 3 5
              2 7 4

# Multiple positions
infinite-wordle guess 1 2 3 4 5 6 7 8
              1 4 1 5 9 2 6 5
```

### 3. Analysis and Solving
```bash
# Analyze position 10
infinite-wordle analyze 10

# Auto-solve with binary search
infinite-wordle solve --strategy binary

# Show game statistics
infinite-wordle stats
```

## Research Extensions

The project structure supports several research directions:

1. **Complexity Analysis**: Add complexity measurement for different strategies
2. **Learning Algorithms**: Implement machine learning approaches
3. **Oracle Machines**: Extend to oracle-computable sequences
4. **Multi-player Version**: Add competitive gameplay

This implementation provides a solid foundation for exploring the mathematical and computational aspects of infinite Wordle variants while maintaining usability and safety.
