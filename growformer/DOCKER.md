# Docker build for Linux (Ubuntu 22.04 LTS)

Produces a Linux amd64 binary for cloud VMs (e.g. 48 vCPU, 96 GB RAM). Same pattern as spacekit: script + Docker.

## Build Linux binary (recommended)

From the `growformer` directory (same pattern as `spacekit.xyz-website-api/build-linux.sh`):

```bash
./build-linux.sh
# or: ./build-linux.sh docker
```

Output: `build/growformer` (Linux amd64). Copy to your Ubuntu 22.04 server and run.

## Build Docker image (run in container)

```bash
docker build -t growformer .
```

## Run MNIST in the container

Mount your MNIST data (decompressed `.ubyte` files) and set `MNIST_ROOT`:

```bash
docker run --rm \
  -e MNIST_ROOT=/data \
  -v /path/to/your/mnist/data:/data \
  growformer --mnist
```

Faster run (fewer samples, fewer epochs, **minibatch** for multi-core):

```bash
docker run --rm \
  -e MNIST_ROOT=/data \
  -v /path/to/your/mnist/data:/data \
  growformer --mnist --mnist-train-limit 2000 --mnist-max-epochs 500 --mnist-batch-size 32
```

Use `--mnist-batch-size 32` or `64` on a many-core machine to parallelize gradient steps (B clones updated in parallel, then params averaged).

## Extract binary to run on the host

If you prefer to run the binary directly on Ubuntu 22.04 (no container):

```bash
docker build -t growformer .
docker create --name gf growformer
docker cp gf:/usr/local/bin/growformer ./growformer-linux
docker rm gf
chmod +x ./growformer-linux
./growformer-linux --mnist
```

## Will 48 CPUs help? Expected speed

- **Minibatch SGD** (`--mnist-batch-size 32` or `64`): Each batch clones the env B times, runs B gradient steps in parallel (one per core), then averages parameters. On 48 cores with e.g. batch size 32 you get **much higher throughput** (many batches per second). Expect **~5–20×** speedup over sequential depending on batch size and core count.
- **Sequential** (no `--mnist-batch-size`): Rayon still parallelizes within each forward/backward (per-layer). You might see **~2–4×** over a small machine, i.e. **~0.1–0.2 epochs/s**.
- **96 GB RAM**: More than enough; minibatch uses B× model memory per batch (e.g. 32× is fine).

Use `--mnist-batch-size 32` (or 64) on your 48-core server for best speed; combine with `--mnist-train-limit` and `--mnist-max-epochs` for quick runs.

---

## EC2 deployment (c4.8xlarge)

**Instance:** c4.8xlarge — 36 vCPU, 60 GiB RAM. Use Ubuntu 22.04 LTS AMI.

### 1. Build the binary (on your Mac or CI)

```bash
cd growformer
./build-linux.sh
```

Binary: `build/growformer` (Linux amd64).

### 2. Copy to EC2

```bash
scp -i your-key.pem build/growformer ubuntu@<EC2-PUBLIC-IP>:~/
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP> chmod +x ~/growformer
```

### 3. MNIST data on the instance

Either download on EC2:

```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
mkdir -p ~/data
cd ~/data
# decompressed .ubyte files required (train-images-idx3-ubyte, train-labels-idx1-ubyte, etc.)
# e.g. wget from a mirror and gunzip, or copy from your machine
```

Or mount/copy your existing `data/` with the four decompressed `.ubyte` files into e.g. `~/data`.

### 4. Run MNIST (36 vCPU → use minibatch 32)

```bash
./growformer --mnist --mnist-batch-size 32
```

With limits for a quicker run:

```bash
./growformer --mnist --mnist-batch-size 32 --mnist-train-limit 2000 --mnist-max-epochs 500
```

If `~/data` holds the MNIST files:

```bash
MNIST_ROOT=~/data ./growformer --mnist --mnist-batch-size 32
```

60 GiB RAM is more than enough; batch size 32 uses ~32× single-model memory and fits easily.

**Running with nohup (survives disconnect):** The progress bar doesn’t animate when stdout/stderr go to a file. Use **`--no-progress`** so epoch lines are printed instead and `tail -f mnist.out` shows progress:
```bash
nohup env MNIST_ROOT=~/data ./growformer --mnist --mnist-batch-size 32 --no-progress > mnist.out 2>&1 &
tail -f mnist.out
```
To see progress when using the bar with nohup, use **`tail -f mnist-run.log`** (log is updated every 50 epochs). To confirm the process is running: `top -p $(pgrep -f growformer)` or `ps aux | grep growformer`.

**If you see "MNIST data not found" on EC2:** Re-run `./deploy-linux.sh` from your Mac — it now fetches MNIST on the server into `~/data` when you don’t have it locally. Or on the server run: `~/download_mnist.sh ~/data` (after a deploy that left the script there).
