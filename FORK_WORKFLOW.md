# Fork layout and syncing

The published repository is https://github.com/kedarkolluri/modded-nanogpt.
This experiment lives entirely under **`experiments/dgx-spark/`** in that fork.
The repository root remains the upstream modded-nanogpt project. Its trainer,
README, dependencies, data tools and record files are not modified by this import.

Run the experiment commands in this directory's README from here:

```sh
cd experiments/dgx-spark
$PY spark/train.py --precision bf16 --compile --fuse-activation \
  --batch 32 --sequence 256 --steps 100 --eval-batches 1 \
  --output results/my-bf16.jsonl
```

FineWeb shards belong in this directory's ignored `data/`, or pass `--data` with
an existing shard directory. For a Spark checkout that reuses our original data:

```sh
$PY spark/train.py --data /home/kedar/projects/inklings2-mini/data \
  --precision mxfp8-native-sr --tail-precision mxfp8 --pack-backend triton-tiled \
  --compile --fuse-activation --batch 32 --sequence 256 --steps 100 \
  --eval-batches 1 --output results/my-mxfp8.jsonl
```

Environments, datasets, binary profiler captures, generated Python caches and
third-party checkouts are ignored. Checked-in reports and CSV/JSON/text evidence
remain available. The original profiler binaries and CUDA environments are kept
in the original Spark workspace, not uploaded to GitHub.

## Sync KellerJordan upstream

From the fork repository root, with your work committed:

```sh
git remote -v
# Once per clone, if this remote does not exist:
git remote add upstream https://github.com/KellerJordan/modded-nanogpt.git
git fetch upstream master
git switch master
git merge --no-edit upstream/master
git push origin master
```

Use a normal merge so fork and upstream history remain connected. Do not reset
or force-push `master` to upstream: that would remove the experiment import.
Keeping our changes inside the experiment directory avoids overlapping edits to
upstream files. Future upstream additions at this same path could still conflict;
Git will report that rather than silently replacing files.

`UPSTREAM_REVISION` records the source used for these experiments. The extracted
ModernArch model in `spark/record_model.py` stays pinned; syncing the upstream
project does not silently change the model behind the benchmark results.

## Bring later changes from the original standalone checkout

The original `/Users/kedarnathkolluri/projects/inklings2-mini` checkout is retained.
Its history was imported with `git subtree add` without squashing. If work is
committed there after this import, bring it into the fork from the fork root:

```sh
git subtree pull --prefix=experiments/dgx-spark \
  /Users/kedarnathkolluri/projects/inklings2-mini main \
  -m "Sync DGX Spark experiment updates"
git push origin master
```

Alternatively, edit files directly under `experiments/dgx-spark/` and commit in
the fork. Avoid making competing edits in both checkouts before a subtree pull;
merge any reported conflicts explicitly. Do not push the standalone checkout's
root tree over the fork's `master` branch.
