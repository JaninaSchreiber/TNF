#!/bin/bash -l
# We changed the M4 comment to d-n-l, not hash
# We need 'bash -l' for the module system

# Copyright 2013 University of Chicago and Argonne National Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# TURBINE-SLURM.SH

# Created: So 29. Nov 09:34:37 CET 2020


# Define convenience macros
# This simply does environment variable substition when m4 runs



#SBATCH --output=/gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2/output.txt
#SBATCH --error=/gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2/output.txt

#SBATCH --partition=main




# TURBINE_SBATCH_ARGS could include --exclusive, --constraint=..., etc.


#SBATCH --job-name=ihr_8000_final2_job

#SBATCH --time=20:59:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --workdir=/gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2

# M4 conditional to optionally perform user email notifications


# User directives:


echo TURBINE-SLURM.SH

export TURBINE_HOME=$( cd "$(dirname "$0")/../../.." ; /bin/pwd )

VERBOSE=0
if (( ${VERBOSE} ))
then
 set -x
fi

TURBINE_HOME=/apps/SWIFTT/1.4.3/turbine
source ${TURBINE_HOME}/scripts/turbine-config.sh

COMMAND="/usr/bin/tclsh8.6 /gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2/swift-t-swift_run_bas.FAZ.tic ihr_8000_final2 ../data/tnf_params.json -ni=20 -nv=3 -st=improving_hit-and-run -cs= -exe=/gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2/spheroid_TNF -settings=/gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2/PhysiCell_settings.xml -ga_parameters=/gpfs/scratch/cns03/cns03363/janina_project/spheroid-tnf-v2-emews/experiments/ihr_8000_final2/tnf_params.json"

# Use this on Midway:
# module load openmpi gcc/4.9

# Use this on Bebop:
# module load icc
# module load mvapich2

TURBINE_LAUNCHER=srun

echo
set -x
${TURBINE_LAUNCHER}  \
                    ${TURBINE_INTERPOSER:-} \
                    ${COMMAND}
# Return exit code from mpirun
