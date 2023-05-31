import os

ROOT_DIR = os.path.join(
    os.path.dirname(__file__),
    "data"
)
os.makedirs(ROOT_DIR, exist_ok=True)