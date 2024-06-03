"""
Loading the best model available.
"""
import lmntfy
import argparse
import asyncio
from pathlib import Path
from lmntfy.models.llm.engine import VllmEngine

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_folder", default="../models", type=Path, help="path to the folder containing all the models")
    parser.add_argument("--debug", default=False, action="store_true", help="Print useful debug information (e.g., prompts)",)
    args = parser.parse_args()
    return args

async def main():
    # process command line arguments
    args = parse_args()
    models_folder = args.models_folder
    verbose = args.debug

    # initializes model
    # NOTE: we do not load a sentence embedder to maximize the GPU memory available
    print("Loading the model...")
    llm = lmntfy.models.llm.Llama3_70b(models_folder, device='cuda', engineType=VllmEngine)

    # chat with the model
    lmntfy.user_interface.command_line.display_logo()
    await lmntfy.user_interface.command_line.basic_chat(llm, verbose=verbose)

if __name__ == "__main__":
    asyncio.run(main())
