# adapted from CleanRL
import time
from dataclasses import dataclass
from typing import Protocol

import gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.buffers import RolloutBuffer

from ccgm.common.algs.core import Agent


class ActorCritic(Protocol):
    actor: nn.Module
    critic: nn.Module


@dataclass
class OnPolicyAgent(
    Agent[
        ActorCritic, RolloutBuffer, torch.optim.Optimizer
    ]
):
    pass


@dataclass
class PPOHparams:
    num_steps: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    update_epochs: int
    norm_adv: bool
    clip_vloss: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float


@dataclass
class PPORparams:
    total_timesteps: int
    batch_size: int
    minibatch_size: int
    num_updates: int


class PPO:

    @staticmethod
    def play(
        agent: OnPolicyAgent,
        envs: gym.vector.VectorEnv,
        hparams: PPOHparams,
        rparams: PPORparams,
        logger,
        global_step: int,
        log_every: int,
        log_file_format: int,
        device: torch.device
    ):
        policy = agent.policy
        rb = agent.memory

        next_obs = rb.observations[rb.pos % rb.buffer_size]
        # interact on policy
        steps = 0
        for step in range(0, hparams.num_steps):
            # TODO: reassigns the same obs at step=0
            steps += 1 * envs.num_envs

            obs = next_obs
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(
                    torch.tensor(next_obs).to(device)
                )

            action = action.cpu().numpy()
            next_obs, reward, done, info = envs.step(action)

            rb.add(obs, action, reward, done, value.flatten(), logprob)

            if logger:
                for item in info:
                    if "episode" in item.keys():
                        logger.add_scalar(
                            "charts/episodic_return",
                            item["episode"]["r"], global_step)
                        logger.add_scalar(
                            "charts/episodic_length",
                            item["episode"]["l"], global_step)
                        break

            if log_every != -1 and global_step % log_every == 0:
                assert log_file_format is not None
                torch.save(
                    agent,
                    log_file_format.format(global_step)
                )

        with torch.no_grad():
            next_obs = torch.tensor(next_obs).to(device)
            next_value = policy.get_value(next_obs).reshape(1, -1)

        rb.compute_returns_and_advantage(
            next_value,
            done
        )

        return steps

    @staticmethod
    def optimize(
        agent: OnPolicyAgent,
        envs: gym.vector.VectorEnv,
        hparams: PPOHparams,
        rparams: PPORparams,
        logger,
        global_step: int,
        log_every: int,
        log_file_format: str,
        device: torch.device
    ):

        if hparams.anneal_lr:
            # Annealing the rate if instructed to do so.
            update = global_step / hparams.num_steps
            frac = 1.0 - (update - 1.0) / rparams.num_updates
            lrnow = frac * hparams.learning_rate
            agent.optimizer.param_groups[0]["lr"] = lrnow

        policy = agent.policy
        optimizer = agent.optimizer
        rb = agent.memory

        # flatten the batch
        b_obs = torch.tensor(rb.observations.reshape(
            (-1,) + envs.single_observation_space.shape)).to(device)
        b_logprobs = torch.tensor(rb.log_probs.reshape(-1)).to(device)
        b_actions = torch.tensor(rb.actions.reshape(
            (-1,) + envs.single_action_space.shape)).to(device)
        b_advantages = torch.tensor(rb.advantages.reshape(-1)).to(device)
        b_returns = torch.tensor(rb.returns.reshape(-1)).to(device)
        b_values = torch.tensor(rb.values.reshape(-1)).to(device)

        # Optimizing the policy and value network
        b_inds = np.arange(rparams.batch_size)
        clipfracs = []
        steps = 0
        for epoch in range(hparams.update_epochs):
            steps += 1
            # construct mini batch
            np.random.shuffle(b_inds)
            for start in range(0, rparams.batch_size, rparams.minibatch_size):
                end = start + rparams.minibatch_size
                mb_inds = b_inds[start:end]

                # compute loss for random mini-batch
                _, newlogprob, entropy, newvalue = policy.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   hparams.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if hparams.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - hparams.clip_coef,
                                1 + hparams.clip_coef)
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
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - hparams.ent_coef * \
                    entropy_loss + v_loss * hparams.vf_coef

                # optimize loss, first applying gradient clipping
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), hparams.max_grad_norm)
                optimizer.step()

            if hparams.target_kl is not None:
                if approx_kl > hparams.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        if logger:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            logger.add_scalar("charts/learning_rate",
                              optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss",
                              pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy",
                              entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl",
                              old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl",
                              approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac",
                              np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance",
                              explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))

        return steps

    @staticmethod
    def learn(
        agent: OnPolicyAgent,
        envs: gym.vector.VectorEnv,
        hparams: PPOHparams,
        rparams: PPORparams,
        logger,
        device,
        log_every: int = -1,  # means no intermediate save log
        log_file_format: str = None
    ):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()

        # num_updates = rparams.total_timesteps // rparams.batch_size

        agent.memory.observations[0] = envs.reset()
        # agent.memory.dones[0] = torch.zeros(envs.num_envs).to(device)
        for update in range(1, rparams.num_updates + 1):
            steps = PPO.play(
                agent=agent,
                envs=envs,
                hparams=hparams,
                rparams=rparams,
                logger=logger,
                global_step=global_step,
                log_every=log_every,
                log_file_format=log_file_format,
                device=device
            )

            global_step += steps

            PPO.optimize(
                agent=agent,
                envs=envs,
                hparams=hparams,
                rparams=rparams,
                logger=logger,
                global_step=global_step,
                log_every=log_every,
                log_file_format=log_file_format
            )

            if logger:
                logger.add_scalar("charts/SPS", int(global_step /
                                (time.time() - start_time)), global_step)

            if agent.memory.full:
                agent.memory.reset()

        return agent

    def predict(
        self,
        agent: OnPolicyAgent,
        obs
    ):
        action, _, _, _ = agent.policy.get_action_and_value(obs)
        return action.cpu().numpy()
