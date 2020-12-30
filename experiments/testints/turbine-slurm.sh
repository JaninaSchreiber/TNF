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

# Created: Mo 14. Dez 15:03:44 CET 2020


# Define convenience macros
# This simply does environment variable substition when m4 runs



# Other key settings are on the sbatch command line
# See turbine-slurm-run.zsh
#SBATCH --time=20:59:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6
#SBATCH --workdir=/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/TNF/experiments/testints

# M4 conditional to optionally perform user email notifications


# User directives:


echo TURBINE-SLURM.SH

export TURBINE_HOME=$( cd "$(dirname "$0")/../../.." ; /bin/pwd )

VERBOSE=0
if (( ${VERBOSE} ))
then
 set -x
fi

TURBINE_HOME=/usr/lib/turbine
source ${TURBINE_HOME}/scripts/turbine-config.sh

COMMAND="/usr/bin/tclsh8.6 /home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/TNF/experiments/testints/swift-t-swift_run_bas.D7F.tic testints ../data/tnf_params.json -ni=20 -nv=3 -st=improving_hit-and-run -cs= -exe=/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/TNF/experiments/testints/spheroid_TNF -settings=/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/TNF/experiments/testints/PhysiCell_settings.xml -ga_parameters=/home/flowerpower/Dokumente/Uni/Barclona/PhysiCell-EMEWS-2-1.0.0/cancer-immune/TNF/experiments/testints/tnf_params.json"

# Use this on Midway:
# module load openmpi gcc/4.9

${TURBINE_LAUNCHER} ${COMMAND}
# Return exit code from mpirun
