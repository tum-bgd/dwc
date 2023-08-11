# On Board Test

This folder contains files for KV260. Please follow these steps to setup:

1. Copy this folder entirely with 

    - `./dwc_ob` (including `.xmodel` and related info)
    - dataset `drone.h5`

    and for testing non-optimized inference efficiency,

    - `trained_weights.pt`
    - `nonopt_isb.py`

    to KV260.

2. Run `ob_opt.py` directly (not in any virtual env) to test the inference speed using the optimized model (`dwc_ob.xmodel`)

3. Setup a virtual environment for testing non-optimized inference efficiency

    ```
    python3 -m venv <env-name>
    ```

    activate

    ```
    source ./<env-name>/bin/activate
    ```

    within the venv, install dependencies:

    ```
    pip install -r requirements.txt
    ```

4. Run `nonopt_isb.py` within venv to test the inference speed using the non-optimized model (`trained-weights.pt`)

Note that we also need to test the inference speed using non-optimized model on common laptops / desktops to get insight into the motivation of this project. But obviously, this would not happen on KV260. You can find the corresponding testing script in the upper-level directory.
