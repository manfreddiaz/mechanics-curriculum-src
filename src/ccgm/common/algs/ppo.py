# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

from argparse import Namespace
from dataclasses import dataclass
import time
from typing import List

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks.p_networks import MlpPolicy
from .networks.v_networks import MlpValueNetwork


@dataclass
class ActorCritic:
    actor: nn.Module
    critic: nn.Module


class PPO:
    def __init__(
        self,
        envs: List,
        rparams,
        hparams,
    ) -> None:
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
        self._envs = envs
        self._hparams = hparams
        self._rparams = rparams 

    def learn(
        self,
        agent: ActorCritic,
        task,
        logger,
        device,
        log_every: int = -1, # means no intermediate save log
        log_file_format: str = None
    ):
        # env setup
        hparams = self._hparams
        rparams = self._rparams

        agent = agent.to(device)
        optimizer = optim.Adam(agent.parameters(), lr=hparams.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((hparams.num_steps, task.num_envs) + self._envs.single_observation_space.shape).to(device)
        actions = torch.zeros((hparams.num_steps, task.num_envs) + self._envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((hparams.num_steps, task.num_envs)).to(device)
        rewards = torch.zeros((hparams.num_steps, task.num_envs)).to(device)
        dones = torch.zeros((hparams.num_steps, task.num_envs)).to(device)
        values = torch.zeros((hparams.num_steps, task.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self._envs.reset()).to(device)
        next_done = torch.zeros(task.num_envs).to(device)
        num_updates = rparams.total_timesteps // rparams.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if hparams.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * hparams.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, hparams.num_steps):
                global_step += 1 * task.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = self._envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        logger.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        logger.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break
                
                if global_step % log_every == 0:
                    assert log_file_format is not None
                    torch.save(
                        agent,
                        log_file_format.format(global_step)
                    )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(hparams.num_steps)):
                    if t == hparams.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + hparams.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + hparams.gamma * hparams.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self._envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self._envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(rparams.batch_size)
            clipfracs = []
            for epoch in range(hparams.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, rparams.batch_size, rparams.minibatch_size):
                    end = start + rparams.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > hparams.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if hparams.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - hparams.clip_coef, 1 + hparams.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if hparams.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -hparams.clip_coef,
                            hparams.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - hparams.ent_coef * entropy_loss + v_loss * hparams.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), hparams.max_grad_norm)
                    optimizer.step()

                if hparams.target_kl is not None:
                    if approx_kl > hparams.target_kl:
                        break

            

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        return agent


    def predict(
        self, 
        agent: ActorCritic,
        obs
    ):
        action, _, _, _ = agent.get_action_and_value(obs)
        return action.cpu().numpy()
