from unsloth import FastVisionModel
import wandb
import os

#region W&B
os.environ["WANDB_API_KEY"]="9d53025453578d5552c59a417fd34da242216859"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "AgriPath-VLM-Sweep"
wandb.login()
#endregion

#region Constants
ARTIFACT_NAME = "model-divine-sweep-5"
SOURCE_ARTIFACT_PATH = "hhm2000-heriot-watt-university/AgriPath-VLM-Sweep/model-divine-sweep-5:v3"
BASE_MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"
#endregion

def main():
    # Initialize a W&B run to use the API
    run = wandb.init(job_type="fix_artifact")

    # 1. Download the existing artifact
    print(f"Downloading artifact: {SOURCE_ARTIFACT_PATH}...")
    artifact = run.use_artifact(SOURCE_ARTIFACT_PATH, type='model')
    artifact_dir = artifact.download()
    print(f"Artifact downloaded to: {artifact_dir}")

    # 2. Load the correct processor from the base model
    print(f"Loading processor from base model: {BASE_MODEL_NAME}...")
    _, processor = FastVisionModel.from_pretrained(model_name=BASE_MODEL_NAME)

    # 3. Save the processor files into the downloaded directory
    print(f"Saving processor to directory: {artifact_dir}...")
    processor.save_pretrained(artifact_dir)
    print("Processor saved successfully.")

    # 4. Create a new artifact with the SAME name to create a new version
    print(f"Creating and logging new version of artifact: {ARTIFACT_NAME}...")
    new_artifact = wandb.Artifact(
        name=ARTIFACT_NAME, # Use the original name
        type="model",
        description="Version 2: Added complete processor configuration."
    )
    new_artifact.add_dir(artifact_dir)
    run.log_artifact(new_artifact)

    run.finish()
    print("\nFix complete! A new version of the artifact has been logged.")
    print(f"The ':latest' tag now points to this new, complete version.")

if __name__ == "__main__":
    main()