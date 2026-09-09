# Recorded execution environments

The final five-seed component runs record Python 3.12.3 and PyTorch
2.6.0+cu124. The final fusion manifest records Python 3.12.3,
scikit-learn 1.8.0, NumPy 2.4.3, and joblib 1.5.3. `requirements.txt`
pins the complete environment used for final component inference, fusion,
analysis, and figure generation. Install the appropriate
PyTorch 2.6.0 CUDA 12.4 wheel from the official PyTorch package index.

The source-work-grouped out-of-fold deep runs and the six-condition matched
module runs separately record Python 3.9.19 and PyTorch 2.1.0; those
accelerator jobs used torch-npu 2.1.0.post6. Their verified probabilities are
included in the confidential companion package. Other package versions from
that accelerator environment were not frozen in the run metadata and are not
invented here. The staged run manifests preserve the recorded Python and
PyTorch versions for every job.

`requirements-test.txt` is an explicit compatibility range for the test
runner; it is not claimed as a package version recorded by a scientific run.
The package audit may use a separate compatible interpreter to execute this
test suite; that test-runner environment is not presented as a scientific run.
