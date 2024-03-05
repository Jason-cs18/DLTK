# Deep Learning Project

Template: https://github.com/Lightning-AI/deep-learning-project-template

## FAIR Hydra (Config Management)
```bash
# File Structure
.
├── conf
│   ├── config.yaml
│── myapp.py
```

```yaml
# conf/config.yaml
db:
  driver: mysql
  user: omry
  pass: secret
```

```python
# myapp.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```