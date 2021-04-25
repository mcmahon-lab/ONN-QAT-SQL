#wandb trial see if works if called from file...
import wandb

def run_wandb():
    for x in range(10):
        wandb.init(project="runs-from-for-loop", reinit=True)
        for y in range (100):
            wandb.log({"metric": x+y})
        wandb.join()