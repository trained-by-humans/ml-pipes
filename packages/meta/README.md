# ml-pipes

Umbrella meta-package for `ml-pipes` component installs.

## Install Profiles

```bash
pip install ml-pipes
pip install 'ml-pipes[vision]'
pip install 'ml-pipes[onnx,vision]'
pip install 'ml-pipes[torch,vision]'
pip install 'ml-pipes[inspection,onnx,vision]'
pip install 'ml-pipes[otel]'
```

The umbrella package installs `ml-pipes-core` by default and uses extras to
pull in the domain packages you need.
