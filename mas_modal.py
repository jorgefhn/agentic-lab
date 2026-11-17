import modal

VOLUME_NAME = "mas-results"
RESULTS_DIR = "/workspace"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets", "accelerate")
    .add_local_python_source("maps_mas_asb")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App("mas-asb")


@app.function(gpu="H100", image=image, timeout=7200, volumes={RESULTS_DIR: volume})
def run_experiment(language: str = "English", topology: str = "chain"):
    from maps_mas_asb import run_full_experiment
    run_full_experiment(language=language, topology=topology, output_dir=RESULTS_DIR)
    volume.commit()


@app.local_entrypoint()
def main():
    run_experiment.remote(language="English", topology="chain")