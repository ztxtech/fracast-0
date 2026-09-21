# Pipeline Template

A minimal starting point for adding a new pipeline flow.

Copy `pipeline.py` to `pipeline/<name>.py`, implement `run(config)`, and
register the flow in `pipeline/pipeline.py`.

## Conventions

- One flow belongs in one module.
- Flow parameters come from the configuration file, not command-line arguments.
- Pipeline files orchestrate model, module, and dataport code; they do not define
  model layers, dataset readers, or command-line interfaces.
- Return metrics in a JSON-serializable dictionary so the experiment runner can
  record them.
