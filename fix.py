from pathlib import Path
import re
import shutil

p = Path("verl/workers/fsdp_workers.py")
source = p.read_text()

start = source.index("    def _normalize_config(self):")
end = source.index("    def normalize_config_stage2", start)

lines = source[start:end].splitlines(keepends=True)

for role in ("rollout", "ref"):
    name = f"self.config.{role}.log_prob_micro_batch_size"

    # 1. 找到原来的整数除法，不依赖缩进或换行格式
    matches = [
        i for i, line in enumerate(lines)
        if re.search(rf"^\s*{re.escape(name)}\s*//=", line)
    ]

    assert len(matches) == 1, (
        f"{role}: 找到 {len(matches)} 处除法，请检查源码"
    )

    i = matches[0]
    j = i

    # 收集完整的多行除法表达式
    while ")" not in lines[j]:
        j += 1
        assert j < len(lines), "除法表达式不完整"

    division = lines[i:j + 1]

    assert "self.device_mesh.shape[0]" in "".join(division)
    assert "self.ulysses_sequence_parallel_size" in "".join(division)

    # 删除原位置的除法
    del lines[i:j + 1]

    # 2. 找到 if/else 分支末尾
    matches = [
        k for k, line in enumerate(lines)
        if re.search(
            rf"^\s*{re.escape(name)}\s*\*=\s*"
            r"self\.config\.rollout\.n_agent\s*$",
            line
        )
    ]

    assert len(matches) == 1, (
        f"{role}: 找不到唯一的 else 分支"
    )

    k = matches[0]
    indent = re.match(r"^(\s*)", division[0]).group(1)

    # 3. 先乘以轨迹扩展倍数，再除以并行设备数
    lines[k + 1:k + 1] = (
        division
        + [
            f'{indent}assert {name} >= 1, '
            f'"{role} micro-batch must be >= 1"\n'
        ]
    )

    print(f"{role}: PATCH_OK")

patched = source[:start] + "".join(lines) + source[end:]

# 写入前进行语法检查
compile(patched, str(p), "exec")

backup = Path(str(p) + ".before_microbatch_fix")
if not backup.exists():
    shutil.copy2(p, backup)

p.write_text(patched)

print("MICROBATCH_FIX_SUCCESS")
