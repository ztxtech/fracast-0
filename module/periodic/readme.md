# periodic/ — 周期/相位编码

## 用途

周期/相位编码。

## 包含什么

- PeriodicPhaseEncoder

## 不包含什么

- 不放上级目录职责内的其它内容（见 `../readme.md` ✓）。
- 不放实验产物（在 `output/` ✓）、不放原始数据（在 `data/` ✓）。

## 命名与组织约定

- 遵循上级目录 `../readme.md` 的约定 ✓。
- 修改本目录代码后，必须跑对应门：模型/组件改动跑**等价性门槛**（`script/verify_model_equivalence.py` ✓），
  流程改动跑**全 pipeline 导入门**（`script/smoke_pipeline_imports.py` ✓）。
