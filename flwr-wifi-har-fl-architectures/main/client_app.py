import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from main.task import load_partitioned_data
from main.task import CNNModel, ResNet50Model, DenseNetModel
from main.task import test as test_fn
from main.task import train as train_fn

# Flower ClientApp
app = ClientApp()

MODEL_DICT = {
    "cnn": CNNModel,
    "resnet50": ResNet50Model,
    "densenet": DenseNetModel,
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = MODEL_DICT[context.run_config["model"]](num_classes=7)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_partitioned_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss, train_acc = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = MODEL_DICT[context.run_config["model"]](num_classes=7)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_partitioned_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

# Simulation code (for analysis.ipynb)
# def set_parameters(net, parameters: List[np.ndarray]):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)


# def get_parameters(net) -> List[np.ndarray]:
#     return [val.cpu().numpy() for _, val in net.state_dict().items()]

# class FlowerClient(NumPyClient):
#     def __init__(self, net, trainloader, valloader):
#         self.net = net
#         self.trainloader = trainloader
#         self.valloader = valloader

#     def get_parameters(self, config):
#         return get_parameters(self.net)

#     def fit(self, parameters, config):
#         set_parameters(self.net, parameters)
#         train_fn(self.net, self.trainloader, epochs=5)
#         return get_parameters(self.net), len(self.trainloader), {}

#     def evaluate(self, parameters, config):
#         set_parameters(self.net, parameters)
#         loss, accuracy = test_fn(self.net, self.valloader)
#         return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

# def client_fn(context: Context) -> Client:
#     """Create a Flower client representing a single organization."""

#     # Load model
#     DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#     model = MODEL_DICT[context.run_config["model" ]](num_classes=7)
#     net = model.to(DEVICE)

#     # Load data (CIFAR-10)
#     # Note: each client gets a different trainloader/valloader, so each client
#     # will train and evaluate on their own unique data partition
#     # Read the node_config to fetch data partition associated to this node
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     batch_size = context.run_config["batch-size"]
#     trainloader, valloader, _ =  trainloader, _ = load_partitioned_data(partition_id, num_partitions, batch_size)

#     # Create a single Flower client representing a single organization
#     # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
#     # to convert it to a subclass of `flwr.client.Client`
#     return FlowerClient(net, trainloader, valloader).to_client()


