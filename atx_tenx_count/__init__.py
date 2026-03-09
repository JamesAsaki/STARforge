"""STARforge package."""

__all__ = ["PipelineConfig", "build_pipeline_config", "run_pipeline"]


def __getattr__(name: str):
    if name in __all__:
        from .pipeline import PipelineConfig, build_pipeline_config, run_pipeline

        exports = {
            "PipelineConfig": PipelineConfig,
            "build_pipeline_config": build_pipeline_config,
            "run_pipeline": run_pipeline,
        }
        return exports[name]
    raise AttributeError(name)
