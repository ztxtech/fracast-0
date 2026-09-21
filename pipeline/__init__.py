"""Pipeline dispatch for corpus preparation and pretraining.

规范：流程放本目录 ✓；模型在 model/、组件在 module/、数据在 dataport/ ✓。
对外只暴露一个入口：`from pipeline import Pipeline` → `Pipeline(config).run()`。
"""
from pipeline.pipeline import Pipeline

__all__ = ["Pipeline"]
