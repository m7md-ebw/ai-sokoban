from stable_baselines3.common.callbacks import BaseCallback

class LiveRenderCallback(BaseCallback):
    def __init__(self, render_every=1000, verbose=0):
        super().__init__(verbose)
        self.render_every = max(1, int(render_every)) if render_every else 0

    def _on_step(self) -> bool:
        if not self.render_every:
            return True
        if self.n_calls % self.render_every != 0:
            return True
        try:
            venv = self.model.get_env()
            raw_env = getattr(venv, "envs", [venv])[0]
            if hasattr(raw_env, "unwrapped"):
                raw_env = raw_env.unwrapped
            raw_env.render(mode="rgb_array")
        except Exception as e:
            if self.verbose:
                print(f"[LiveRender] render failed: {e}")
        return True

def _extract_success(info):
    if isinstance(info, (list, tuple)) and len(info) > 0:
        info = info[0]
    for k in ("all_goals_satisfied", "all_boxes_on_target", "is_success"):
        v = info.get(k, False)
        if isinstance(v, (bool, int)) and bool(v):
            return True
    return False

class SuccessEvalCallback(BaseCallback):
    """
    Runs a quick evaluation every eval_freq steps and logs eval/success_rate to TensorBoard.
    """
    def __init__(self, eval_env, eval_freq=50_000, n_eval_episodes=10, max_steps=500, deterministic=True, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.max_steps = int(max_steps)
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or (self.n_calls % self.eval_freq) != 0:
            return True
        successes = 0
        for ep in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = [False]
            steps = 0
            ep_success = False
            while not done[0] and steps < self.max_steps:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = self.eval_env.step(action)
                if _extract_success(info):
                    ep_success = True
                steps += 1
            successes += int(ep_success)
        rate = successes / self.n_eval_episodes
        
        self.model.logger.record("eval/success_rate", rate)
        if self.verbose:
            print(f"[Eval] success_rate={rate:.2%} ({successes}/{self.n_eval_episodes}) at {self.num_timesteps} steps")
        return True
