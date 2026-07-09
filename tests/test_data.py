from mlops_lab.data import load_dataset, sample_rows


def test_split_sizes():
    splits = load_dataset("breast_cancer", test_size=0.2, seed=42)
    total = len(splits.X_train) + len(splits.X_test)
    assert total == 569  # full breast cancer dataset
    assert abs(len(splits.X_test) / total - 0.2) < 0.01


def test_split_deterministic():
    a = load_dataset("breast_cancer", test_size=0.2, seed=42)
    b = load_dataset("breast_cancer", test_size=0.2, seed=42)
    assert list(a.X_test.index) == list(b.X_test.index)


def test_different_seed_differs():
    a = load_dataset("breast_cancer", test_size=0.2, seed=42)
    b = load_dataset("breast_cancer", test_size=0.2, seed=43)
    assert list(a.X_test.index) != list(b.X_test.index)


def test_no_train_test_leakage():
    splits = load_dataset("breast_cancer", test_size=0.2, seed=42)
    assert set(splits.X_train.index).isdisjoint(splits.X_test.index)


def test_unknown_dataset():
    import pytest

    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("nope")


def test_sample_rows():
    rows = sample_rows("breast_cancer", n=5)
    assert len(rows) == 5
