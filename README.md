# HPC Workshop — Introductory Course Examples

Hands-on materials for the RTU HPC introductory course. The five tasks progress from submitting a first PBS job to GPU rendering, CUDA programming, MPI-based numerical solvers, and large-scale neural network training.

# Table of contents

- [Course overview](#course-overview)
- [Contents of this repository](#contents-of-this-repository)
- [Software prerequisites](#software-prerequisites)
- [Accessing the RTU HPC cluster](#accessing-the-rtu-hpc-cluster)
- [Getting the examples](#getting-the-examples)
- [Task descriptions](#task-descriptions)
  - [Task 1 — Simple PBS job](#task-1--simple-pbs-job)
  - [Task 2 — Parametric job: Blender animation rendering](#task-2--parametric-job-blender-animation-rendering)
  - [Task 3 — CUDA hello world](#task-3--cuda-hello-world)
  - [Task 4 — Homework: train a toy GPT](#task-4--homework-train-a-toy-gpt)
  - [Task 5 — MPI heat solver](#task-5--mpi-heat-solver)

# Course overview

This workshop covers the full HPC usage stack:

- **Job scheduling** — describing and submitting computational tasks to a PBS (Torque) cluster
- **Parametric jobs** — running the same program over many independent inputs simultaneously (task arrays)
- **GPU computing** — writing and running CUDA kernels; requesting GPU nodes
- **Distributed memory parallelism** — splitting a numerical problem across nodes with MPI
- **Large-scale ML** — training a Transformer model using cluster GPU resources

# Contents of this repository

```
workshop/
├── task1/
│   ├── hostname.sh              # PBS job script — single-node hello world
│   └── hostname_parallel.sh    # PBS job script — MPI multi-node hostname
│
├── task2/
│   ├── test.blend              # Input: Blender scene to render
│   ├── run_blender.sh          # PBS array job script (100 frames)
│   └── check_result.sh         # Validates rendered output files
│
├── task3_cuda/
│   ├── hello-world.cu          # CUDA kernel source
│   └── run_cuda.sh             # PBS job script (requests GPU node)
│
├── task4_homework/
│   ├── toy_GPT.py              # GPT-style Transformer implementation
│   ├── shakespeare_full.txt    # Full training corpus (~5 MB)
│   └── shakespeare_small.txt   # Reduced corpus for quick tests
│
└── task5_mpi/
    ├── bottle.dat              # Input: geometry and initial conditions
    ├── run_heat.sh             # PBS job script — parallel heat solver
    ├── heat_solver.pdf         # Numerical method documentation
    └── labs_c/                 # C source code for the heat solver
```

# Software prerequisites

All software is available as environment modules on the cluster. Load them as shown in each task.

| Software | Module name | Tasks |
|---|---|---|
| OpenMPI | `openmpi` | 1 (parallel), 5 |
| Blender | `blender/2.70` | 2 |
| CUDA toolkit | `cuda/cuda-12.4` | 3 |
| PyTorch (GPU) | `AI/pytorch-2.5.1-gpu-sg` | 4 |

List available versions of any package:

```bash
module avail <package_name>
```

# Accessing the RTU HPC cluster

Connection instructions (web terminal, SSH, credentials) are available in the official guide:

**[Connect to the HPC cluster](https://hpc-platforma.rtu.lv/hpc.html#connect-to-the-hpc-cluster)**

After login you are placed in your home directory `/home/<username>`. Scientific software is installed under `/opt/exp_soft/`.

## Common job management commands

| Action | Command |
|---|---|
| Submit a job | `qsub job_script.sh` |
| List your jobs | `qstat` or `qstat -u $USER` |
| List array sub-jobs | `qstat -t` |
| Check free resources | `showq` |
| Check running jobs | `showq -r` |
| Cancel a job | `qdel <JOB_ID>` |
| Interactive session | `qsub -I` |
| Interactive session with GPU | `qsub -l nodes=1:ppn=1:gpus=1 -I` |

# Getting the examples

**Option A — copy from the cluster shared directory:**

```bash
cp -r ~/job_examples/workshop ./
cd workshop
```

**Option B — clone from GitHub:**

```bash
git clone https://github.com/rtuhpc/workshop.git
cd workshop
```

---

# Task descriptions

## Task 1 — Simple PBS job

**Goal:** submit your first job and verify it executes on a compute node.

`hostname.sh` requests a single CPU core and prints the hostname of the allocated node:

```bash
echo "Hello world from node $HOSTNAME"
echo "Sveiciens no nodes $HOSTNAME"
```

`hostname_parallel.sh` loads OpenMPI and runs `/bin/hostname` across all allocated nodes using `mpirun`, showing how a single job spans multiple machines.

### PBS directives used

```bash
#PBS -N simple_test
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=1,mem=1g
#PBS -l feature=epyc
#PBS -q batch
#PBS -j oe
```

### Running

```bash
# Single node
qsub task1/hostname.sh

# Multi-node via MPI
qsub task1/hostname_parallel.sh
```

Job output appears in your working directory as `<jobname>.o<JOB_ID>`.

To run interactively instead:

```bash
qsub -I
echo "Sveiciens no nodes `hostname`"
exit
```

To target a specific node:

```bash
qsub -l nodes=wn68 task1/hostname.sh
```

---

## Task 2 — Parametric job: Blender animation rendering

**Goal:** render a 100-frame animation by distributing one frame per CPU core using a PBS array job.

Each frame takes roughly 1 minute on a single CPU, so 100 frames would take ~1h40min sequentially. With a PBS array job, all frames run in parallel (up to 50 at a time), completing in approximately 1 minute per frame regardless of total count.

`run_blender.sh` submits a job array where the environment variable `$PBS_ARRAYID` carries the frame number to each sub-job:

```bash
#PBS -t 1-100%50       # 100 sub-jobs, max 50 concurrent
#PBS -N blendertests

frame=$PBS_ARRAYID
blender -b test.blend -o rend### -F JPEG -t 2 -f $frame
```

### Running

```bash
cd task2
qsub run_blender.sh

# Monitor array progress
qstat -t

# Verify all 100 frames rendered
./check_result.sh <JOB_ID>
```

### Assemble frames into an animation

```bash
convert -delay 20 rend*.jpg blender_out.gif
```

### Copy results to your local machine

**Linux / macOS:**

```bash
scp <username>@<login-node>:workshop/task2/blender_out.gif ./
```

**Windows:** use [WinSCP](https://winscp.net/eng/download.php) with the same server address and your login credentials.

---

## Task 3 — CUDA hello world

**Goal:** write and run a minimal CUDA kernel to understand how GPU threads execute in parallel.

`hello-world.cu` sends the string `"Hello "` and an offset array to a GPU kernel. Each thread adds its offset to the corresponding character's ASCII value, producing `"World!"`. The result is copied back to the host and printed.

```
h(104) + 15 = w(119)
e(101) + 10 = o(111)
l(108) +  6 = r(114)  →  "Hello " + kernel = "World!"
...
```

To request a GPU node, add to your PBS script:

```bash
#PBS -l nodes=1:ppn=1:gpus=1,feature=l40s
```

### Running

```bash
cd task3_cuda
qsub run_cuda.sh
```

### Manual compile and run (on a GPU node or local workstation with CUDA)

```bash
module load cuda/cuda-12.4
nvcc hello-world.cu -o out
./out
```

---

## Task 4 — Homework: train a toy GPT

**Goal:** use the HPC cluster with GPUs to train and experiment with a small Transformer (LLM) model.

`toy_GPT.py` implements a GPT-style character-level language model using PyTorch. It is trained on Shakespeare's works and generates new text after training. Your task is to get it running on the cluster, experiment with the parameters, and report on the results.

| Component | Detail |
|---|---|
| Dataset | `shakespeare.txt` (~5 MB) |
| Tokenizer | GPT-2 BPE via `tiktoken` |
| Framework | PyTorch |
| Model | Custom small Transformer |

### Hints

- Starting point: `toy_GPT.py` in `task4_homework/`
- Pre-built PyTorch modules available on the cluster:
  - `AI/pytorch-1.13.1-gpu-conda`
  - `AI/pytorch-2.5.1-gpu-sg`
- Monitor GPU utilization: [job efficiency guide](https://hpc-guide.rtu.lv/07_job_managment.html#job-efficiency)
- For multi-GPU training, look into PyTorch Distributed Data Parallel (DDP)
- Feel free to use ChatGPT or other resources for help and examples

### Deliverable (for team submission, due mid-April)

- Python code used
- GPU utilization screenshot
- Sample of generated text
- Conclusions on model quality vs. training parameters

---

## Task 5 — MPI heat solver

**Goal:** solve the 2-D heat equation in parallel across multiple CPU cores using MPI and visualise the result.

The solver decomposes the 2-D grid across MPI ranks. Each rank computes its sub-region and exchanges boundary values with neighbours at every time step. A pre-compiled `heat` binary is included — `labs_c/` contains the C source for reference.

### Running locally (on the login node, for testing)

```bash
module load openmpi
./heat                        # default 256×256 grid, 500 iterations
./heat bottle.dat             # bottle geometry, 500 iterations
./heat bottle.dat 1000        # bottle geometry, 1000 iterations
./heat 1024 2048 1000         # explicit grid size, 1000 iterations
```

### Submitting to the cluster

`run_heat.sh` uses `$PBS_NODEFILE` and `$PBS_NUM_PPN` to distribute ranks automatically:

```bash
qsub -l nodes=1:ppn=8 run_heat.sh
```

To find the optimal core count, vary `ppn` and compare wall-clock times (strong scaling experiment):

```bash
qsub -l nodes=4:ppn=12 run_heat.sh
```

### Assemble output frames into an animation

```bash
convert -delay 20 *.png heat.gif
```

Refer to `heat_solver.pdf` for a description of the finite-difference scheme, boundary conditions, and convergence criteria.

---

## Modification suggestions

| Task | What to try |
|---|---|
| Task 1 | Increase node count; observe `mpirun` distribute across them |
| Task 2 | Render more frames; change the `%50` concurrency limit |
| Task 3 | Increase thread/block count; modify the kernel operation |
| Task 4 | Tune Transformer depth, heads, context length; compare CPU vs GPU speed |
| Task 5 | Plot wall-clock time vs. core count to produce a scaling curve |
