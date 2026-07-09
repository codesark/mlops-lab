from mlops_lab.config import ChampionConfig, Direction, GateSpec
from mlops_lab.evaluate import check_gates, compare_to_champion


class TestCheckGates:
    def test_pass(self):
        result = check_gates({"rmse": 0.5}, {"rmse": GateSpec(max=0.6)})
        assert result.passed and not result.failures

    def test_exactly_at_threshold_passes(self):
        assert check_gates({"rmse": 0.6}, {"rmse": GateSpec(max=0.6)}).passed
        assert check_gates({"r2": 0.75}, {"r2": GateSpec(min=0.75)}).passed

    def test_max_violation(self):
        result = check_gates({"rmse": 0.7}, {"rmse": GateSpec(max=0.6)})
        assert not result.passed
        assert "above max" in result.failures[0]

    def test_min_violation(self):
        result = check_gates({"r2": 0.5}, {"r2": GateSpec(min=0.75)})
        assert not result.passed
        assert "below min" in result.failures[0]

    def test_multiple_failures_aggregate(self):
        result = check_gates(
            {"rmse": 0.9, "r2": 0.1},
            {"rmse": GateSpec(max=0.6), "r2": GateSpec(min=0.75)},
        )
        assert len(result.failures) == 2

    def test_missing_metric_fails(self):
        result = check_gates({}, {"rmse": GateSpec(max=0.6)})
        assert not result.passed
        assert "not computed" in result.failures[0]


MINIMIZE = ChampionConfig(primary_metric="rmse", direction=Direction.minimize, tolerance=0.02)
MAXIMIZE = ChampionConfig(primary_metric="f1", direction=Direction.maximize, tolerance=0.01)


class TestChampionComparison:
    def test_bootstrap_no_champion(self):
        result = compare_to_champion({"rmse": 0.5}, None, MINIMIZE)
        assert result.passed
        assert "bootstrap" in result.reason

    def test_champion_missing_metric_passes(self):
        result = compare_to_champion({"rmse": 0.5}, {"other": 1.0}, MINIMIZE)
        assert result.passed
        assert "bootstrap" in result.reason

    def test_candidate_missing_metric_fails(self):
        result = compare_to_champion({"other": 1.0}, {"rmse": 0.5}, MINIMIZE)
        assert not result.passed

    def test_minimize_better_passes(self):
        assert compare_to_champion({"rmse": 0.45}, {"rmse": 0.5}, MINIMIZE).passed

    def test_minimize_within_tolerance_passes(self):
        # 0.5 * 1.02 = 0.51 is the limit
        assert compare_to_champion({"rmse": 0.509}, {"rmse": 0.5}, MINIMIZE).passed

    def test_minimize_beyond_tolerance_fails(self):
        assert not compare_to_champion({"rmse": 0.52}, {"rmse": 0.5}, MINIMIZE).passed

    def test_maximize_better_passes(self):
        assert compare_to_champion({"f1": 0.97}, {"f1": 0.96}, MAXIMIZE).passed

    def test_maximize_within_tolerance_passes(self):
        # 0.96 * 0.99 = 0.9504 is the limit
        assert compare_to_champion({"f1": 0.951}, {"f1": 0.96}, MAXIMIZE).passed

    def test_maximize_beyond_tolerance_fails(self):
        assert not compare_to_champion({"f1": 0.94}, {"f1": 0.96}, MAXIMIZE).passed
