"""STARforge package."""

__all__ = ["PipelineConfig", "run_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import PipelineConfig, run_pipeline

        exports = {
            "PipelineConfig": PipelineConfig,
            "run_pipeline": run_pipeline,
        }
        return exports[name]
    raise AttributeError(name)
