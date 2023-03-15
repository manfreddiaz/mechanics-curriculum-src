# adapted from CleanRL

from dataclasses import dataclass, astuple
import time

import gym
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ActorCritic:
    actor: nn.Module
    critic: nn.Module


@dataclass
class OnPolicyReplayBuffer:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __iter__(self):
        fields = [
            self.obs, self.actions, self.logprobs, 
            self.rewards, self.dones, self.values, 
            self.advantages, self.returns]
        for field in fields:
            yield field
    
@dataclass
class OnPolicyAgent:
    policy: ActorCritic
    memory: OnPolicyReplayBuffer
    optimizer: torch.optim.Optimizer


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


class PPO:
    
    @staticmethod
    def play(
        agent: OnPolicyAgent,
        next_obs: torch.Tensor,
        next_done: torch.Tensor,
        envs: gym.vector.SyncVectorEnv,
        hparams: PPOHparams,
        rparams: PPORparams,
        logger,
        global_step: int,
        log_every,
        log_file_format,
    ):

        obs, actions, logprobs, rewards, dones, values, advantages, returns = agent.memory
        policy = agent.policy
        # interact on policy
        for step in range(0, hparams.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = policy.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(rewards.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(obs.device), torch.Tensor(done).to(dones.device)

            for item in info:
                if "episode" in item.keys():
                    # print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    logger.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    logger.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break
            
            if log_every != -1 and global_step % log_every == 0:
                assert log_file_format is not None
                torch.save(
                    agent,
                    log_file_format.format(global_step)
                )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = policy.get_value(next_obs).reshape(1, -1)
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
                # TODO: verify that this step is equivalent to the previous
                # I did it to make the update straight in the agent memory
                returns[t] = advantages[t] + values[t]

        return next_obs, next_done

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
    ):
        
        
        policy = agent.policy
        optimizer = agent.optimizer 

        obs, actions, logprobs, _, _, values, advantages, returns = agent.memory
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(rparams.batch_size)
        clipfracs = []
        for epoch in range(hparams.update_epochs):
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

                # optimize loss, first applying gradient clipping 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), hparams.max_grad_norm)
                optimizer.step()

            if hparams.target_kl is not None:
                if approx_kl > hparams.target_kl:
                    break
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger:
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

        return agent

    @staticmethod
    def learn(
        agent: OnPolicyAgent,
        envs: gym.vector.VectorEnv,
        hparams: PPOHparams,
        rparams: PPORparams,
        logger,
        device,
        log_every: int = -1, # means no intermediate save log
        log_file_format: str = None
    ):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
       
        num_updates = rparams.total_timesteps // rparams.batch_size
        
        next_obs = torch.Tensor(envs.reset()).to(device)
        next_done = torch.zeros(envs.num_envs).to(device)
        for update in range(1, num_updates + 1):
            if hparams.anneal_lr:
                # Annealing the rate if instructed to do so.
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * hparams.learning_rate
                agent.optimizer.param_groups[0]["lr"] = lrnow
            
            next_obs, next_done = PPO.play(
                agent=agent,
                next_obs=next_obs,
                next_done=next_done,
                envs=envs,
                hparams=hparams,
                rparams=rparams,
                logger=logger,
                global_step=global_step,
                log_every=log_every,
                log_file_format=log_file_format
            )

            global_step += 1 * envs.num_envs * hparams.num_steps

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
            
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


        return agent
            
    def predict(
        self, 
        agent: OnPolicyAgent,
        obs
    ):
        action, _, _, _ = agent.policy.get_action_and_value(obs)
        return action.cpu().numpy()
