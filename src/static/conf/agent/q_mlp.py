
import logging
import numpy as np
import torch.nn as nn

from ccgm.common.algs.networks.q_networks import MlpQNetwork

log = logging.getLogger(__name__)


def make_agent(id: str):
    log.info(f"<create> agent {id}")
    return MlpQNetwork
