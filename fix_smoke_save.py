from pathlib import Path
import re
import shutil

p = Path("verl/trainer/ppo/ray_trainer_ts.py")
s = p.read_text()

flag = "self.config.trainer.get('smoke_skip_saving', False)"

replacements = [
    (
        "if metrics['critic/rewards/mean'] > self.best_reward:",
        f"if not {flag} and metrics['critic/rewards/mean'] > self.best_reward:"
    ),
    (
        "if 'val/test_score/mean' in metrics and metrics['val/test_score/mean'] > self.best_val:",
        f"if not {flag} and 'val/test_score/mean' in metrics and metrics['val/test_score/mean'] > self.best_val:"
    ),
    (
        "                if self.global_steps >= self.total_training_steps:\n",
        "                if self.global_steps >= self.total_training_steps:\n"
        f"                    if {flag}:\n"
        "                        print('SMOKE_STEP_FINISHED_NO_CHECKPOINT', flush=True)\n"
        "                        return\n"
    )
]

for old, new in replacements:
    assert s.count(old) == 1, f"匹配失败或重复: {old}"
    s = s.replace(old, new, 1)

compile(s, str(p), "exec")

shutil.copy2(p, str(p) + ".before_smoke_save_fix")
p.write_text(s)

# 创建专用 smoke 脚本，保留原 lowram 脚本
src = Path("experiments/r2_tree_rl_step1_v100_lowram.sh")
dst = Path("experiments/r2_tree_rl_step1_v100_nosave.sh")

shell = src.read_text()
pattern = re.compile(r"(?P<q>['\"]?)trainer\.save_freq=-1(?P=q)")
matches = list(pattern.finditer(shell))

assert len(matches) == 1, "未找到唯一的 trainer.save_freq=-1"

m = matches[0]
q = m.group("q")
new_arg = q + "+trainer.smoke_skip_saving=true" + q
shell = shell[:m.start()] + new_arg + " " + shell[m.start():]

dst.write_text(shell)
dst.chmod(src.stat().st_mode)

print("SMOKE_CHECKPOINT_FIX_OK")
print("Created:", dst)
