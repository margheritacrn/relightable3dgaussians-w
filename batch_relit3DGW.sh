#!/bin/bash
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=margherita.lea.corona@hhi.fraunhofer.de
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2

#####################################################################################

# This included file contains the definition for $LOCAL_JOB_DIR to be used locally on the node.
source "/etc/slurm/local_job_dir.sh"

# The next line is optional and for job statistics only. You may omit it if you do not need statistics. 
echo "$PWD/${SLURM_JOB_ID}_stats.out" > $LOCAL_JOB_DIR/stats_file_loc_cfg

# Make a folder locally on the node for job_results. This folder ensures that data is copied back even when the job fails 
mkdir -p "${LOCAL_JOB_DIR}/job_results"




# Launch the apptainer image with --nv for nvidia support. Two bind mounts are used: 
# - One for the ImageNet dataset ($DATAPOOL1/datasets/ImageNet-complete) mapped to /mnt/dataset in the container and 
# - One for the results (e.g. checkpoint data that you may store in $LOCAL_JOB_DIR on the node
# That locations are passed as parameters ( /mnt/dataset/train /mnt/dataset/val /mnt/output) to the program in the run-section of the container 
# Check the line in the runsection (python -m resnet_50 train "$@") which runs the training and "$@" has all the parameters and passes them on to the python program

echo "Run full eval on NeRF-OSR dataset"
apptainer run --nv --bind $DATAPOOL3/datasets/nerfosr:/mnt/dataset --bind ${LOCAL_JOB_DIR}/job_results:/mnt/output ./relit3DGS-W.sif --output_path /mnt/output --nerfosr /mnt/dataset

# These commands copy all results generated in $LOCAL_JOB_DIR back to the submit folder regarding the job id.
cd "$LOCAL_JOB_DIR"
tar -cf zz_${SLURM_JOB_ID}.tar job_results 
cp zz_${SLURM_JOB_ID}.tar $SLURM_SUBMIT_DIR
rm -rf ${LOCAL_JOB_DIR}/job_results