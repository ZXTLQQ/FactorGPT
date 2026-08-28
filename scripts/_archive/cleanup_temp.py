# -*- coding: utf-8 -*-
"""清理本次数据下载任务的一次性临时脚本（保留对运维有长期价值的）。"""
import os

SCRIPTS = r"e:/factor-gpt/scripts"
keep = {"check_export_status.py", "final_check.py", "probe_library.py",
        "verify_miner_load.py", "compile_all.py", "check_garbled_dirs.py",
        "cleanup_entry.py", "normalize_registry.py"}
remove = {"verify_local_channel.py", "verify_docx.py", "smoke_eval.py",
          "verify_offline_data.py", "write_history_readme.py", "check_cndata_range.py",
          "bench_batch.py"}
for name in remove:
    p = os.path.join(SCRIPTS, name)
    if os.path.exists(p):
        os.remove(p)
        print("REMOVED:", name)
print("\nscripts dir now:", sorted(os.listdir(SCRIPTS)))
