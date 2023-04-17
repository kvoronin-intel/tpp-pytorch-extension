
Prerequsite:
-----------

Install and activate conda environment as described in README.md in the repo (TODO add link).

Install task specific requirements (one time):
$pip install -r requirements.txt

Put correct paths to LIBXSMM, LIBXSMM-DNN and libiomp5.so into run_nompi_template.sh

To create an initial checkpoint:
#bash run_nompi_template.sh create_start_checkpoint <batch_size>

To run single-socket (and no data loaders) bf16 training with TPP/ParLoopER:
#bash run_nompi_template.sh run_bf16_training <batch_size>

Note: batch_size should be 56 for SPR and 64 for GVT3 (x16.vlarge) 64-core instance
