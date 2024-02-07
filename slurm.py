import os
from datetime import datetime
import argparse
import time
import socket
import subprocess
import yaml
import wandb
import numpy as np

def get_gpu_info(gpu_type=['a6000', 'a5000'], remove_nodes=None):
    def run(cmd, print_err=True):
        try:
            return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('UTF-8').splitlines()
        except subprocess.CalledProcessError as e:
            # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            if print_err:
                print("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
            return [cmd.split()[-1]]
    gpudata = run('sinfo -O nodehost,gres -h')
    new_gpu_data = []
    for gpu in gpu_type:
        new_gpu_data += [line.split(' ')[0] for line in gpudata if gpu in line]
    new_gpu_data = set(new_gpu_data)
    if remove_nodes is not None:
        remove_nodes = set(remove_nodes)
        new_gpu_data = new_gpu_data - remove_nodes
    new_gpu_data = list(new_gpu_data)

    assert len(new_gpu_data) > 0, 'No GPU found'
    return ','.join(new_gpu_data)
    
    

qos_dict = {"sailon" : {"nhrs" : 2, "cores": 16, "mem":128},
            "scav" : {"nhrs" : 24, "cores": 16, "mem":128},
            "vulc_scav" : {"nhrs" : 24, "cores": 16, "mem":128},
            "cml_scav" : {"nhrs" : 24, "cores": 16, "mem":128}, 

            "high" : {"gpu":4, "cores": 16, "mem":120, "nhrs": 36},
            "medium" : {"gpu":2, "cores": 8, "mem":64, "nhrs": 72},
            "default" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168},
            "tron" : {"gpu":1, "cores": 4, "mem":32, "nhrs": 168}}




def check_qos(args):
    
    for key, max_value in qos_dict[args.qos].items():
        val_from_args = getattr(args, key)
        if val_from_args != None:
            if val_from_args > max_value:
                raise ValueError("Invalid paramter for {} for {}".format(key, args.qos))
        else:
            setattr(args, key, max_value)
    return args


#TODO: Add day funtionality too 
parser = argparse.ArgumentParser()
parser.add_argument('--nhrs', type=int, default=24)
parser.add_argument('--base-dir', default=f'{os.getcwd()}')
parser.add_argument('--output-dirname', default='output')
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--qos', default="high", type=str, help='Qos to run')
parser.add_argument('--env', default="reinforce_cosine_all_boundaries", type=str, help = "Set the name of the dir you want to dump")
parser.add_argument('--gpu', default=4, type=int, help='Number of gpus')
parser.add_argument('--cores', default=16, type=int, help='Number of cpu cores')
parser.add_argument('--mem', default=120, type=int, help='RAM in G')




parser.add_argument('--middle', action='store_true')

# parser.add_argument('--batch_size', default=64, type=int, help='Batch size')



gpu_types = ['a6000']
remove_nodes = ['cml23', 'tron61', 'tron56', 'cml21']

# parser.add_argument('--path', default='/fs/vulcan-projects/actionbytes/vis/ab_training_run3_rerun_32_0.0001_4334_new_dl_nocasl_checkpoint_best_dmap_ab_info.hkl')
# parser.add_argument('--num_ab', default= 100000, type=int, help='number of actionbytes')

args = parser.parse_args()

nodes = get_gpu_info(gpu_types,remove_nodes)



args = parser.parse_args()
time_str = str(int(time.time()))
args.env += str(int(time.time()))


output_dir = os.path.join(args.base_dir, args.output_dirname, args.env)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



print("Output Directory: %s" % output_dir)
step = 1

    #adding 


params= [(config) for config in ['ssv2_small/MoLo_SSv2_Small_1shot_v1.yaml',
                                 'ssv2_full/MoLo_SSv2_Full_1shot_v1.yaml']]


port_start = min(int(np.random.uniform()*10e4), 65000)
output_dir = f'{args.base_dir}/output/{args.env}/'
with open(f'{args.base_dir}/output/{args.env}/now.txt', "w") as nowfile,\
     open(f'{args.base_dir}/output/{args.env}/log.txt', "w") as output_namefile,\
     open(f'{args.base_dir}/output/{args.env}/err.txt', "w") as error_namefile,\
     open(f'{args.base_dir}/output/{args.env}/name.txt', "w") as namefile:

    for i, (config) in enumerate(params):
        now = datetime.now()
        master_port = port_start + i
        wandb_id = wandb.util.generate_id()
        datetimestr = now.strftime("%m%d_%H%M:%S.%f")
        name = f'test_{i}'

        cmd = f'python runs/run.py --cfg configs/projects/MoLo/{config}'
        
        nowfile.write(f'{cmd}\n')
        namefile.write(f'{(os.path.join(output_dir, name))}.log\n')
        output_namefile.write(f'{(os.path.join(output_dir, name))}_log.txt\n')
        error_namefile.write(f'{(os.path.join(output_dir, name))}_error.txt\n')
        #break
###########################################################################
# Make a {name}.slurm file in the {output_dir} which defines this job.
#slurm_script_path = os.path.join(output_dir, '%s.slurm' % name)
start=1
slurm_script_path = os.path.join(output_dir, f'test.slurm')
slurm_command = "sbatch %s" % slurm_script_path

# Make the .slurm file
with open(slurm_script_path, 'w') as slurmfile:
    slurmfile.write("#!/bin/bash\n")
    slurmfile.write(f"#SBATCH --array=1-{len(params)}\n")
    #slurmfile.write(f"#SBATCH --array=1-10\n")
    slurmfile.write("#SBATCH --output=/dev/null\n")
    slurmfile.write("#SBATCH --error=/dev/null\n")
    slurmfile.write("#SBATCH --requeue\n")
    slurmfile.write("#SBATCH --nodes=1\n")
    # slurmfile.write("#SBATCH --exclude=vulcan[00-23]\n")

    
    args = check_qos(args)


    if "scav" in args.qos or "tron" in args.qos:
        if args.qos == "scav":
            slurmfile.write("#SBATCH --account=scavenger\n")
            slurmfile.write("#SBATCH --qos scavenger\n")


        elif args.qos == "vulc_scav":
            slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
            slurmfile.write("#SBATCH --qos vulcan-scavenger\n")
            slurmfile.write("#SBATCH --partition vulcan-scavenger\n")
        elif args.qos == 'cml_scav':
            slurmfile.write("#SBATCH --account=cml-abhinav\n")
            slurmfile.write("#SBATCH --qos cml-scavenger\n")
       
        
        slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
        slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
        slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)

        if not args.gpu is None:
            # if hostname in {'nexus', 'vulcan'}:
            
            slurmfile.write(f'#SBATCH --gres=gpu:{args.gpu}\n')
        else:
            raise ValueError("Specify the gpus for scavenger")
            
    elif args.qos == 'high':
            slurmfile.write("#SBATCH  --qos vulcan-high\n")
            slurmfile.write("#SBATCH --partition vulcan-ampere\n")
            slurmfile.write("#SBATCH --account=vulcan-abhinav\n")
            slurmfile.write("#SBATCH --time=%d:00:00\n" % args.nhrs)
            slurmfile.write("#SBATCH --gres=gpu:%d\n" % args.gpu)
            slurmfile.write("#SBATCH --cpus-per-task=%d\n" % args.cores)
            slurmfile.write("#SBATCH --mem=%dG\n" % args.mem)
    slurmfile.write("#SBATCH --nodelist=%s\n" % nodes)
    
    
    
    slurmfile.write("\n")
    #slurmfile.write("export MKL_SERVICE_FORCE_INTEL=1\n")p
    slurmfile.write("cd " + os.getcwd() + '\n')
    slurmfile.write("module load ffmpeg\n")
    slurmfile.write("export MKL_THREADING_LAYER=GNU\n")
    slurmfile.write("source /fs/cfar-projects/actionloc/new_miniconda/bin/activate\n")
    slurmfile.write("conda activate pips2\n")
    slurmfile.write(f"srun --output=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/log.txt | tail -n 1) --error=$(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/err.txt | tail -n 1)  $(head -n $SLURM_ARRAY_TASK_ID {args.base_dir}/output/{args.env}/now.txt | tail -n 1)\n")
    slurmfile.write("\n")
print(slurm_command)
print("Running on {}, with {} gpus, {} cores, {} mem for {} hour {}".format(args.qos, args.gpu, args.cores, args.mem , args.nhrs, port_start))
if not args.dryrun:
   os.system("%s &" % slurm_command)
