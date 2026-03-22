import argparse

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from main.task import load_centralized_dataset, test
from main.task import CNNModel, ResNet50Model, DenseNetModel


# Create ServerApp
app = ServerApp()

MODEL_DICT = {
    "cnn": CNNModel,
    "resnet50": ResNet50Model,
    "densenet": DenseNetModel,
}

def compute_model_size_bytes(arrays: ArrayRecord) -> int:
    """Compute the model size in bytes."""
    return sum(t.numel() * t.element_size() for t in arrays.to_torch_state_dict().values())

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = MODEL_DICT[context.run_config["model"]](num_classes=7)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    
    #Compute communication overhead per round
    num_clients = len(grid.get_node_ids())
    model_size = compute_model_size_bytes(arrays)
    comm_per_round = 2 * num_clients * model_size
    total_comm = comm_per_round * num_rounds
    print(f"Model size: {model_size/1e6:.2f} MB")
    print(f"Clients: {num_clients}")
    print(f"Communication per round: {comm_per_round/1e6:.2f} MB")
    print(f"Total communication for {num_rounds} rounds: {total_comm/1e6:.2f} MB\n")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=lambda rnd, arr: global_evaluate(rnd, arr, global_model),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord, model: torch.nn.Module) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})