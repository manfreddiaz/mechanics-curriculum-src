import os
import yaml

from ccgm.common.envs.sl.mnist.config import ROOT_DIR


with open(os.path.join(ROOT_DIR, 'players.yaml'), mode='r') as f:
    SPURIOUS : dict = yaml.safe_load(f)