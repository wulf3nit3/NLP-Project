# SETUP.md - what to type when you connect

Every block below says **where** it runs. "Login node" is whatever `c-00X` machine you land on
after ssh; it is not the same as a compute node, which is where jobs actually run.

## Every session

**Where:** your local machine, then the cluster login node. One command per line, in order.

```bash
ssh <tau-username>@slurm-client.cs.tau.ac.il
```

```bash
bash
```

```bash
export PROJECT_ROOT="/home/morg/NLP_2526b/$USER"
source "$PROJECT_ROOT/LMEnt/cluster/start.sh"
```

- `bash` is not optional - TAU logs you into tcsh, where none of this works. It is a separate
  command because you have to be in bash *before* the next line is readable.
- **`source`, not `bash`.** Running the file instead starts a second shell, sets everything up
  there, and throws it away - your own shell is left untouched. The script refuses to run that way
  rather than appearing to work.
- It changes directory to the checkout, loads the `lment_*` commands, and turns on conda. You can
  be anywhere when you source it; it finds the repository from its own path. Once you are already
  in the checkout, `source cluster/start.sh` is the same thing with less typing.
- Add `--no-env` to skip conda. Submitting jobs does not need it - the job scripts activate their
  own environment on the compute node. Running python yourself does.

If you would rather do it by hand, or something in the script misbehaves, that is exactly these
four steps:

```bash
export PROJECT_ROOT="/home/morg/NLP_2526b/$USER"
cd "$PROJECT_ROOT/LMEnt"
```

```bash
source cluster/lmentrc.sh
```

```bash
lment_env
```

Then `lment_help` for the command list, or `lment_where` if you've lost track of which machine
you're on.

## Run the experiment

**Where:** cluster login node, in `$PROJECT_ROOT/LMEnt`, after sourcing `start.sh`.

```bash
bash slurm/run_sweep.sh 64000
bash slurm/run_sweep.sh 128000
bash slurm/run_sweep.sh 256000
```

Use `lment_jobs` to watch the queue and `lment_log` to inspect the latest job log.
When a wave finishes, score it with `slurm/score_ladder.sh`.

## When something breaks

| What you see | What it means |
|---|---|
| Pasted command → syntax errors | You're in tcsh. Type `bash`. |
| `tmux attach`: no such session | tmux only exists on one machine. `hostname` first, then `ssh c-00X`. |
| `python: No such file or directory` in a job | The conda env is half-built. Rebuild it on a **login** node. |
