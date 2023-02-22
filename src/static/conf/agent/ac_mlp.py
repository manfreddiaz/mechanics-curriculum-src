
import logging
from ccgm.common.algs.networks.ac_network import MlpActorCritic

log = logging.getLogger(__name__)

def make_agent(
    id: str
):
    log.info(f'building agent: {id}')
    return MlpActorCritic
