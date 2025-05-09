# predict_cli.py

import pickle
import numpy as np
import typer
from rich import print
from rich.panel import Panel
from rich.prompt import Prompt
from pathlib import Path

app = typer.Typer(
    help="🏠 House‐price predictor (runs entirely in your terminal)"
)

FEATURES = [
    ("bedrooms", int),
    ("bathrooms", float),
    ("sqft_living", float),
    ("sqft_lot", float),
    ("floors", float),
    ("view", int),
    ("condition", int),
    ("grade", int),
    ("yr_built", int),
    ("yr_renovated", int),
]

def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def select_model_file(models_dir: Path) -> Path:
    """List all .pkl files in models_dir and let the user pick one."""
    models_dir = models_dir.expanduser()
    files = sorted(models_dir.glob("*.pkl"))
    if not files:
        typer.secho(f"❌ No model files found in {models_dir}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # If there's only one, pick it without prompting
    if len(files) == 1:
        return files[0]

    # Otherwise, show a dropdown choice
    choices = [f.name for f in files]
    choice = Prompt.ask("Select model", choices=choices, default=choices[0])
    return models_dir / choice

@app.command()
def predict(
    model_path: Path = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Full path to a tree_model.pkl; if omitted you'll get a dropdown of outputs/*.pkl"
    ),
    bedrooms: int = typer.Option(..., prompt="Number of bedrooms"),
    bathrooms: float = typer.Option(..., prompt="Number of bathrooms"),
    sqft_living: float = typer.Option(..., prompt="Living area (sqft)"),
    sqft_lot: float = typer.Option(..., prompt="Lot size (sqft)"),
    floors: float = typer.Option(..., prompt="Number of floors"),
    view: int = typer.Option(..., prompt="View rating (0–4)"),
    condition: int = typer.Option(..., prompt="Condition rating (1–5)"),
    grade: int = typer.Option(..., prompt="Grade rating (1–13)"),
    yr_built: int = typer.Option(..., prompt="Year built"),
    yr_renovated: int = typer.Option(..., prompt="Year renovated"),
):
    """
    Run the tree‐based model on the features you supply.
    If you omit --model-path, you'll choose one interactively.
    """
    # If no model_path passed, let user pick from outputs/*.pkl
    if model_path is None:
        outputs_dir = Path(__file__).parent / "outputs"
        model_path = select_model_file(outputs_dir)

    if not model_path.exists():
        typer.secho(f" Model file not found at {model_path!r}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    model = load_model(model_path)
    x = np.array([[
        bedrooms, bathrooms, sqft_living, sqft_lot,
        floors, view, condition, grade,
        yr_built, yr_renovated
    ]])
    price = model.predict(x)[0]

    print()
    print(Panel(f"[bold green]${price:,.2f}[/]", title=f"Predicted Sale Price\n(model: {model_path.name})", expand=False))
    print()

if __name__ == "__main__":
    app()
