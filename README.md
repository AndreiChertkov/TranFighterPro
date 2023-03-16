# TranFighterPro

> Black-box adversarial attack on machine translation systems.


## Installation

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name TranFighterPro python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate TranFighterPro
    ```

4. Install dependencies:
    ```bash
    pip install jupyterlab teneva==0.13.0 ttopt==0.5.0 protes==0.1.2 torch torchvision scikit-image matplotlib jupyterlab "jax[cpu]==0.4.3" optax transformers
    ```

5. Set your own code for translator requests inside function `_translate` in the script `tranopti.py` (see also comments in the script).

6. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name TranFighterPro --all -y
    ```


## Usage

1. Run the main script `python tranopti.py [LEVEL] [ENGINE]`
    > The results will be in the `results/ENGINE` folder. The `LEVEL` may be 1-3 (it relates to number of words in the input text for translation), and `ENGINE` may be `deepl`, `google`, `yandex` or `all`.

> Note that the screens of performed requests to translators (`deepl`, `google`, `yandex`) are available here https://disk.yandex.ru/d/wCrlv6BH8sktng


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Ivan Oseledets](https://github.com/oseledets)
