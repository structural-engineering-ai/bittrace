import pytest
import os
from tests.generate_sample_dataset import generate_dataset

@pytest.fixture(scope="session", autouse=True)
def setup_sample_dataset():
    """
    Automatically generate synthetic dataset for tests before any test runs.
    """
    base_path = "./tests/sample_data"
    if not os.path.exists(base_path):
        print("Generating synthetic sample dataset for tests...")
        generate_dataset(base_path=base_path, digits=range(10), samples_per_class=5)
    yield
    # Optional teardown can go here if needed
