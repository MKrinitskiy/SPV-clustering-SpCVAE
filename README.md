# SPV-clustering-SpCVAE
Codebase for the study "[Clustering of Polar Vortex States Using Convolutional Autoencoders](https://ceur-ws.org/Vol-2426/paper8.pdf)" by Mikhail Krinitskiy, Yulia Zyulyaeva and Sergey Gulev.



```latex
@inproceedings{krinitskiy12019clustering,
  author = {Mikhail Krinitskiy, Yulia Zyulyaeva and Sergey Gulev},
  title = {Clustering of polar vortex states using convolutional autoencoders},
  year = {2019}
}

@proceedings{ithpc2019,
  editor =       {Sergey I. Smagin and Alexander A. Zatsarinnyy},
  title =        "Information Technologies and High-Performance Computing",
  booktitle =    "Short Paper Proceedings of the V International Conference on Information Technologies and High-Performance Computing",
  publisher =    {ceur-ws.org}
  venue =        {Khabarovsk, Russia},
  month =        sep,
  year =         {2019}
}
```



Usage:

```bash
$ train.sh 
```



`train.py` arguments one may specify in `train.sh`:

`--snapshot` - switch for resuming training from a snapshot.

`--run-name` - name for a training run, which is used for naming of logs and backups directories

`--batch-size` - batch size, `default=32`

`--val-batch-size` - batch size for validation stage, `default=32`

`--gpu` - id of the GPU to use (as reported by `nvidia-smi` )

`--epochs` - number of epochs to train; `default=200`

`--steps-per-epoch` - number of steps (batches) per epoch

`--val-steps` - number of steps (batches) per validation stage; `default=100`

`--no-snapshots` - the switch disabling snapshots saving

`--variational` - the switch enabling variational loss component

`--debug` - the switch enabling `DEBUG` mode

`--embeddims` - embeddings dimensionality; `default=128`
