"""
Create the GraphLIME Reimplementation project structure.
"""

from pathlib import Path

PROJECT_NAME = "graphlime-reimplementation"

folders = [
    "configs",
    "data",
    "datasets",
    "models",
    "graphlime",
    "scripts",
    "evaluation",
    "visualization",
    "tests",
    "outputs/checkpoints",
    "outputs/figures",
    "outputs/explanations",
    "notebooks",
]

files = {
    "README.md": "# GraphLIME Reimplementation\n",
    ".gitignore": "",
    "LICENSE": "",
    "requirements.txt": "",
    "setup.py": "",
    "configs/gcn.yaml": "",
    "configs/graphsage.yaml": "",
    "configs/graphlime.yaml": "",
    "datasets/__init__.py": "",
    "datasets/loader.py": "",
    "datasets/preprocessing.py": "",
    "models/__init__.py": "",
    "models/gcn.py": "",
    "models/graphsage.py": "",
    "models/trainer.py": "",
    "graphlime/__init__.py": "",
    "graphlime/kernel.py": "",
    "graphlime/hsic.py": "",
    "graphlime/lasso.py": "",
    "graphlime/explainer.py": "",
    "graphlime/utils.py": "",
    "evaluation/__init__.py": "",
    "evaluation/fidelity.py": "",
    "evaluation/sparsity.py": "",
    "evaluation/metrics.py": "",
    "visualization/__init__.py": "",
    "visualization/node_plot.py": "",
    "visualization/feature_bar.py": "",
    "visualization/graph_visualizer.py": "",
    "scripts/train.py": "",
    "scripts/explain.py": "",
    "scripts/evaluate.py": "",
    "scripts/visualize.py": "",
    "tests/test_kernel.py": "",
    "tests/test_hsic.py": "",
    "tests/test_explainer.py": "",
    "notebooks/Demo.ipynb": "",
}


def main():
    root = Path(PROJECT_NAME)
    root.mkdir(exist_ok=True)

    for folder in folders:
        (root / folder).mkdir(parents=True, exist_ok=True)

    for file, content in files.items():
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    print("=" * 60)
    print("GraphLIME project created successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
