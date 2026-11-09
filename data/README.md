This directory is place to find your data.

If you have local dataset please move it here. By default `.csv` are ignored from commits to not commit large datasets into github. 
If you want to add different file type, please add it as exception to `.gitignore` file like `data/*.csv`.

Everything under `data/` is gitignored (`data/*` in `.gitignore`, plus explicit
`data/*.npz|csv|rar|zip`), except this README.

The shared datasets live in `../datasets` (i.e. `DCL/datasets`), nested as
`<name>(current)/<name>/`. They are linked here as symlinks so the notebooks'
`data/<name>` paths and the experiment scripts' default data root (`./data`)
resolve without copying ~1 GB:

```
data/first_batch  -> ../../datasets/first_batch(current)/first_batch
data/other_data   -> ../../datasets/other_data(current)/other_data
data/second_batch -> ../../datasets/second_batch(current)/second_batch   # appears once that batch is available
```

Recreate with: `cd data && ln -sfn "../../datasets/<name>(current)/<name>" <name>`.
On the HPC, `jobs/pre.sh` links `FASTODT_DATA_SHARED/*/` into `data/` the same way.

