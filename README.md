# mlops-lab

End-to-end MLOps simulation: notebook development → tested package → CI quality
gates → MLflow Model Registry → manager-approved (mocked) deployment.

Full architecture docs coming with the notebooks. Quick start:

```bash
brew install libomp          # macOS only, for xgboost
uv sync
make test
make mlflow-up DEV=1         # local throwaway MLflow at http://127.0.0.1:5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
make train-all
```
