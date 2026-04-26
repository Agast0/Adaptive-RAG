"""AdaptiveRAG+ additive module.

Adds class-D global retrieval on top of Adaptive-RAG (A/B/C) without modifying
any existing code paths. Public sub-modules:

  contracts        : Internal record schema and routing/metric contracts.
  longbench        : LongBench (gov_report, qmsum) ingestion + Class-D labelling.
  global_index     : Offline chunk/embed/cluster/summarize/index pipeline.
  global_inference : Online retrieval + generation with efficiency tracing.
  routing          : 4-way (A/B/C/D) router built on existing classifier outputs.
  evaluation       : Per-task metric synthesis (EM/F1/Acc/Step/Time and ROUGE-L).
"""

__all__ = [
    "contracts",
    "longbench",
    "global_index",
    "global_inference",
    "routing",
    "evaluation",
]
