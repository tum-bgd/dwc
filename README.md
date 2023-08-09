# classification on drone image tiles

- four classes: debris, forest, water, other

- compile command (on host for on-board depolyment):
    ```bash
    vai_c_xir -x ./quantize_result/Model_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json -o dwc_ob -n dwc_ob
    ```