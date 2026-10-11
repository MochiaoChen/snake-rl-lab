"""Double DQN with immediate collision action masks, including replay targets."""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN


class GlobalDQN(DQN):
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        obs = np.asarray(observation)
        vectorized = obs.ndim > 1
        batch = obs if vectorized else obs[None, :]
        valid = batch[:, -3:] > .5
        with th.no_grad():
            tensor, _ = self.policy.obs_to_tensor(batch)
            q = self.q_net(tensor)
            mask = th.as_tensor(valid, device=self.device)
            actions = q.masked_fill(~mask, -1e9).argmax(1).cpu().numpy()
        if not deterministic:
            for i in range(len(actions)):
                if np.random.random() < self.exploration_rate:
                    actions[i] = np.random.choice(np.flatnonzero(valid[i]))
        return (actions if vectorized else actions[0]), state

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        actions, buffer_actions = super()._sample_action(learning_starts, action_noise, n_envs)
        # Warm-up random actions also respect the mask.
        for i in range(n_envs):
            valid = self._last_obs[i, -3:] > .5
            if not valid[int(actions[i])]:
                actions[i] = np.random.choice(np.flatnonzero(valid))
                buffer_actions[i] = actions[i]
        return actions, buffer_actions

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                valid = data.next_observations[:, -3:] > .5
                # Online network selects; target network evaluates (Double DQN).
                next_actions = self.q_net(data.next_observations).masked_fill(~valid, -1e9).argmax(1, keepdim=True)
                next_q = self.q_net_target(data.next_observations).gather(1, next_actions)
                discounts = data.discounts if data.discounts is not None else self.gamma
                target = data.rewards + (1 - data.dones) * discounts * next_q
            current = self.q_net(data.observations).gather(1, data.actions.long())
            loss = F.smooth_l1_loss(current, target)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
