# Adapted from Zama Concrete-ML MNIST example code.
# Original source: https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/use_case_examples/mnist
# Local modifications: timing/reporting/reproducibility utilities.

import argparse
import warnings
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import torch



# Concrete-Numpy and Concrete-ML
from concrete.numpy.compilation import Configuration

# The QAT model
from model import CommonQuant, MNISTQATModel
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model


def train(model, device, train_loader, optimizer, epoch, criterion):
    """Train the model."""

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    len(train_loader.dataset) // len(data),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, criterion):
    """Test the model."""

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        "Accuracy: "
        f"{correct}/{len(test_loader.dataset)} ({100.0 * correct / len(test_loader.dataset):.0f}%)"
    )

    return test_loss


def manage_dataset(train_kwargs, test_kwargs, data_dir):
    """Get training and test parts of MNIST dataset."""

    # Pre-transform
    class ReshapeTransform:
        def __init__(self, new_size):
            self.new_size = new_size

        def __call__(self, img):
            return torch.reshape(img, self.new_size)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ReshapeTransform((28 * 28,)),
        ]
    )

    # Manage datasets
    dataset1 = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def compile_and_test(
    model,
    use_virtual_lib,
    np_inputs,
    quantization_bits,
    test_data,
    test_data_length,
    test_target,
    show_mlir,
    current_index,
):
    # Compile the QAT model and test
    configuration = Configuration(
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
        use_insecure_key_cache=True,  # This is for our tests only, never use that in prod
        insecure_key_cache_location="/tmp/keycache",
        p_error=None,  # To avoid any confusion: we are always using kwarg p_error
    )

    if use_virtual_lib:
        print(f"\n{current_index}. Compiling with the Virtual Library")
    else:
        print(f"\n{current_index}. Compiling in FHE")

    q_module = compile_torch_model(
        model,
        np_inputs,
        import_qat=True,
        configuration=configuration,
        # Note that in CML 0.4, fixing net_inputs and net_outputs to 5 will no more be needed,
        # since it will be the default
        n_bits={
            "net_inputs": 5,
            "op_inputs": quantization_bits,
            "op_weights": quantization_bits,
            "net_outputs": 5,
        },
        use_virtual_lib=use_virtual_lib,
        show_mlir=show_mlir,
    )

    # Check max bit width
    max_bit_width = q_module.forward_fhe.graph.maximum_integer_bit_width()

    if max_bit_width > 8:
        raise Exception(
            f"Too large bit-width ({max_bit_width}): training this network resulted in an "
            "accumulator size that is too large. Possible solutions are:"
            "    - this network should, on average, have 8bit accumulators. In your case an unlucky"
            f"initialization resulted in {max_bit_width} accumulators. You can try to train the "
            "network again"
            "    - reduce the sparsity to reduce the number of active neuron connexions"
            "    - if the weight and activation bitwidth is more than 2, you can try to reduce one "
            "or both to a lower value"
        )

    # Check the accuracy
    if use_virtual_lib:
        print(
            f"\n{current_index + 1}. Checking accuracy with the Virtual Library "
            f"(length {test_data_length})"
        )
    else:
        print(f"\n{current_index + 1}. Checking accuracy in FHE (length {test_data_length})")

    # Key generation
    if not use_virtual_lib:
        q_module.forward_fhe.keygen()

    correct_fhe = 0

    # Reduce the test data, since very slow in FHE
    reduced_test_data = test_data[0:test_data_length, :]

    encrypt_run_decrypt_times = []

    for idx, im in enumerate(tqdm(reduced_test_data)):
        target_np = test_target[idx]
        q_data = q_module.quantize_input(im)
        q_data = np.expand_dims(q_data, 0).astype(np.int64)

        encrypt_run_decrypt_start = time.perf_counter()
        prediction = q_module.forward_fhe.encrypt_run_decrypt(q_data)
        encrypt_run_decrypt_times.append(time.perf_counter() - encrypt_run_decrypt_start)
        prediction = q_module.dequantize_output(prediction)

        if np.argmax(prediction) == target_np:
            correct_fhe += 1

    average_encrypt_run_decrypt_seconds = float(np.mean(encrypt_run_decrypt_times))

    # Final accuracy
    return (
        correct_fhe,
        reduced_test_data.shape[0],
        max_bit_width,
        average_encrypt_run_decrypt_seconds,
    )


def save_experiment_report(settings, results, model_path, reports_dir):
    """Save an experiment report and model snapshot with a descriptive file name."""

    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        f"mnist_q{settings['quantization_bits']}"
        f"_s{settings['sparsity']}"
        f"_seed{settings['seed']}"
        f"_nr{int(bool(settings['narrow_range']))}"
        f"_fhe{settings['test_data_length_reduced']}"
        f"_{timestamp}"
    )

    model_snapshot_path = reports_dir / f"{base_name}.onnx"
    if model_path.exists():
        shutil.copy2(model_path, model_snapshot_path)
    else:
        model_snapshot_path = None

    report = {
        "timestamp": timestamp,
        "settings": settings,
        "results": results,
        "model_path": str(model_path.resolve()) if model_path.exists() else None,
        "model_snapshot_path": str(model_snapshot_path.resolve()) if model_snapshot_path else None,
    }

    report_path = reports_dir / f"{base_name}.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nSaved experiment report to: {report_path.resolve()}")
    if model_snapshot_path:
        print(f"Saved model snapshot to: {model_snapshot_path.resolve()}")


def parse_args(argv=None):
    """Parse reproducibility settings without requiring source edits."""

    parser = argparse.ArgumentParser(
        description="Train and evaluate the MNIST QAT-FCNN baseline with Concrete-ML."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--sparsity",
        type=int,
        choices=(4, 6, 8, 10, 11, 12),
        default=4,
        help="Multiplier for 14 active inputs per neuron; the paper uses all six choices.",
    )
    parser.add_argument("--quantization-bits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3108559580)
    parser.add_argument("--fhe-samples", type=int, default=5)
    parser.add_argument("--full-test-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.33)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training when available.")
    parser.add_argument("--show-mlir", action="store_true")
    parser.add_argument(
        "--no-training",
        action="store_true",
        help="Evaluate the supplied ONNX model instead of training a new model.",
    )
    parser.add_argument(
        "--wide-range",
        action="store_true",
        help="Disable the narrow-range quantization used for the paper comparison.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("mnist.qat.onnx"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("experiment_reports"))
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 1 <= args.fhe_samples <= 10000:
        parser.error("--fhe-samples must be between 1 and 10000")
    if not 1 <= args.full_test_samples <= 10000:
        parser.error("--full-test-samples must be between 1 and 10000")
    return args


def main(argv=None):
    """Main."""

    args = parse_args(argv)
    warnings.filterwarnings("ignore")

    np.set_printoptions(threshold=1024)
    criterion = nn.CrossEntropyLoss()
    torch.autograd.set_detect_anomaly(True)

    # Options: the most important ones
    epochs = args.epochs
    sparsity = args.sparsity
    quantization_bits = args.quantization_bits
    do_test_in_fhe = True
    do_training = not args.no_training
    show_mlir = args.show_mlir

    # Options: can be changed
    lr = args.learning_rate
    gamma = args.gamma
    test_data_length_reduced = args.fhe_samples
    test_data_length_full = args.full_test_samples

    # Options: no real reason to change
    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    use_cuda_if_available = args.cuda
    seed = args.seed
    model_path = args.model_path
    CommonQuant.narrow_range = not args.wide_range

    # Seeding
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)

    print(f"\nUsing seed {seed}\n")
    torch.manual_seed(seed)

    # Training and test arguments
    train_kwargs = {"batch_size": batch_size}
    test_kwargs = {"batch_size": test_batch_size}

    # Cuda management
    use_cuda = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Manage dataset
    train_loader, test_loader = manage_dataset(train_kwargs, test_kwargs, args.data_dir)

    # Model definition
    model = MNISTQATModel(quantization_bits, quantization_bits)
    model = model.to(device)
    model.prune(sparsity, True)

    # Start
    print(
        f"Performing MNIST task with {quantization_bits}-bits in quantization and a "
        f"sparsity of {sparsity}"
    )

    if do_training:
        print("\n1. Training")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        test_loss = 1e10

        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch, criterion)
            test(model, device, test_loader, criterion)

            scheduler.step()

        model.prune(sparsity, False)

        # Export to ONNX
        print("\n2. Exporting to ONNX")
        dummy_input = torch.rand((1, 784)).to(device)
        torch.onnx.export(model, dummy_input, model_path, opset_version=14)

    else:
        print("\n1. Loading pre-trained model")
        if not model_path.is_file():
            raise FileNotFoundError(f"Pre-trained ONNX model not found: {model_path}")

    # Reload the model
    model = onnx.load(model_path)

    # Test in FHE
    if do_test_in_fhe:

        list_inputs = []

        for inputs in test_loader:
            inputs_var, targets = inputs
            list_inputs.append(inputs_var.detach().cpu().numpy())

        np_inputs = np.concatenate(list_inputs, axis=0)

        test_data = np.zeros((len(test_loader.dataset), 784))
        test_target = np.zeros((len(test_loader.dataset), 1))
        idx = 0

        for data, target in tqdm(test_loader):
            target_np = target.cpu().numpy()
            for idx_batch, im in enumerate(data.numpy()):
                test_data[idx] = im
                test_target[idx] = target_np[idx_batch]
                idx += 1

        accuracy = {}
        bit_widths = {}
        timing = {}
        current_index = 3

        for use_virtual_lib, use_full_dataset in [(True, True), (True, False), (False, False)]:
            test_data_length = (
                test_data_length_full if use_full_dataset else test_data_length_reduced
            )

            (
                correct_fhe,
                test_data_shape_0,
                max_bit_width,
                average_encrypt_run_decrypt_seconds,
            ) = compile_and_test(
                model,
                use_virtual_lib,
                np_inputs,
                quantization_bits,
                test_data,
                test_data_length,
                test_target,
                show_mlir,
                current_index,
            )

            current_index += 2
            current_accuracy = correct_fhe / test_data_shape_0

            print(
                f"Accuracy in {'VL' if use_virtual_lib else 'FHE'} with length {test_data_length}: "
                f"{correct_fhe}/{test_data_shape_0} = "
                f"{current_accuracy:.4f}, in {max_bit_width} bits"
            )

            if not use_virtual_lib:
                print(
                    f"Average encrypt_run_decrypt time over {test_data_shape_0} samples: "
                    f"{average_encrypt_run_decrypt_seconds:.2f}s"
                )

            if (use_virtual_lib, use_full_dataset) == (True, True):
                accuracy["VL full"] = current_accuracy
                bit_widths["VL full"] = max_bit_width
            elif (use_virtual_lib, use_full_dataset) == (True, False):
                accuracy["VL short"] = current_accuracy
                bit_widths["VL short"] = max_bit_width
            else:
                assert (use_virtual_lib, use_full_dataset) == (False, False)
                accuracy["FHE short"] = current_accuracy
                bit_widths["FHE short"] = max_bit_width
                timing["fhe_encrypt_run_decrypt_avg_seconds"] = average_encrypt_run_decrypt_seconds
                timing["fhe_samples"] = test_data_shape_0

    # Check that accuracy in FHE and in VL is the same
    assert (
        accuracy["VL short"] == accuracy["FHE short"]
    ), "Error, accuracy in VL and in FHE are not the same"

    # Check that accuracy is random-looking
    assert accuracy["VL full"] > 0.8, "Error, accuracy is too bad"

    experiment_settings = {
        "epochs": epochs,
        "sparsity": sparsity,
        "active_connections": 14 * sparsity,
        "quantization_bits": quantization_bits,
        "do_test_in_fhe": do_test_in_fhe,
        "do_training": do_training,
        "show_mlir": show_mlir,
        "lr": lr,
        "gamma": gamma,
        "test_data_length_reduced": test_data_length_reduced,
        "test_data_length_full": test_data_length_full,
        "batch_size": batch_size,
        "test_batch_size": test_batch_size,
        "use_cuda_if_available": use_cuda_if_available,
        "seed": seed,
        "narrow_range": CommonQuant.narrow_range,
    }

    experiment_results = {
        "accuracy": accuracy,
        "bit_widths": bit_widths,
        "timing": timing,
    }
    save_experiment_report(
        experiment_settings,
        experiment_results,
        model_path=model_path,
        reports_dir=args.reports_dir,
    )

    print()


if __name__ == "__main__":
    main()
