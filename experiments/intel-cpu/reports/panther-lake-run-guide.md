# Panther Lake run guide (Windows and Linux)

This guide runs experiment A (and later B-E) on Intel client PCs. The
measurement is only valid when the machine state is controlled and recorded.

## 0. Expected CPU properties (verify with `env_report.py`)

Panther Lake (Core Ultra series 3) mixes P-cores, E-cores and low-power
E-cores. Expected ISA: AVX2, FMA, AVX-VNNI (and related VEX extensions);
**no AVX-512 and no AMX**. Do not trust this list: `env.json` records the real
flags (`cpu.isa_flags_os` on Linux, `cpu_features_numpy` on both OSes) and the
core types.

## 1. One-time setup

### Both OSes

1. Install git and `uv` (<https://docs.astral.sh/uv/>).
2. Clone the fork and check out the branch:

   ```text
   git clone https://github.com/mrsabhar/docling
   cd docling
   git checkout intel-cpu-exp
   ```

3. Make sure the machine can reach `huggingface.co` and `www.modelscope.cn`
   once (model download). After the first run you can set
   `HF_HUB_OFFLINE=1` to forbid further downloads.
4. Python 3.12 is used; `uv` downloads it if it is not installed.

### Windows

```powershell
winget install --id astral-sh.uv -e   # or the installer from docs.astral.sh/uv
```

- Exclude the repo folder and `%LOCALAPPDATA%\uv` from Microsoft Defender
  real-time scanning, or record that you did not. Scanning inflates import time
  and cold start.
- Long paths: `git config --global core.longpaths true`.

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install linux-tools-common linux-tools-$(uname -r)   # cpupower (optional)
```

## 2. Machine state before every run

| Item | Windows | Linux |
|---|---|---|
| Power source | AC adapter connected (also run once on battery, label it) | same |
| Power profile | Settings -> System -> Power & battery -> Power mode = **Best performance**. Record OEM mode (vendor utility) if any. `powercfg /getactivescheme` is recorded automatically. | `powerprofilesctl set performance`, or `sudo cpupower frequency-set -g performance` and EPP `performance` (see below). Governor, EPP and driver are recorded automatically. |
| Battery saver / Energy saver | Off | n/a |
| Background load | Close apps, pause OneDrive / Windows Update / indexing | Stop heavy services; no desktop compositor load if possible |
| Thermal | Let the machine idle 5 min before the run; same surface and lid state every time | same |
| Foreground | Keep the terminal window in the foreground. Windows can treat background processes as low QoS (EcoQoS) and move them to E-cores. | n/a |

Linux EPP (intel_pstate active mode):

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
```

## 3. Identify core types

### Linux

```bash
cat /sys/devices/cpu_core/cpus    # P-cores, e.g. 0-3
cat /sys/devices/cpu_atom/cpus    # E-cores and LP E-cores, e.g. 4-15
lscpu -e=CPU,CORE,CACHE,MAXMHZ    # LP E-cores: lowest MAXMHZ, no shared L3 id
```

### Windows

- Sysinternals Coreinfo: `coreinfo64 -c -l` lists cores and caches. Task Manager
  -> Performance -> CPU (right click -> Change graph to -> Logical processors)
  also shows the count.
- PowerShell, core and logical CPU counts (no core type):

  ```powershell
  Get-CimInstance -Namespace root/cimv2 -ClassName Win32_Processor |
      Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
  ```

  For the per-core efficiency class, use Coreinfo or Intel's
  `Intel Processor Identification Utility`. Record the P-core logical CPU
  numbers; you need them for the affinity mask.

## 4. Run matrix for experiment A

Run each row. Use the label exactly, so results from different machines and
OSes stay separate. `<sku>` is a short machine name, for example `ptl-u7-355`.

| # | Label | Thread setting | Affinity |
|---|---|---|---|
| A1 | `<sku>-<os>-ac-default` | none (Docling default `num_threads=4`) | none |
| A2 | `<sku>-<os>-ac-pcores` | `DOCLING_NUM_THREADS` = P-core count | P-cores only |
| A3 | `<sku>-<os>-ac-allcores` | `DOCLING_NUM_THREADS` = all physical cores | none |
| A4 | `<sku>-<os>-dc-default` | none | none, **on battery** |

A1 is the baseline that later experiments compare with. A2-A4 show how much
the default leaves on the table and feed experiment D.

### Linux commands

```bash
# A1
experiments/intel-cpu/run_bench.sh --label ptl-u7-355-linux-ac-default

# A2 (P-cores 0-3 in this example; child processes inherit the affinity)
DOCLING_NUM_THREADS=4 taskset -c 0-3 \
  experiments/intel-cpu/run_bench.sh --label ptl-u7-355-linux-ac-pcores

# A3 (16 cores in this example)
DOCLING_NUM_THREADS=16 experiments/intel-cpu/run_bench.sh --label ptl-u7-355-linux-ac-allcores
```

### Windows commands (PowerShell)

```powershell
# A1
.\experiments\intel-cpu\run_bench.ps1 -Label ptl-u7-355-win-ac-default

# A2: P-cores are logical CPUs 0-3 in this example -> mask 0xF.
# Child processes inherit the affinity mask of the process that starts them.
$env:DOCLING_NUM_THREADS = "4"
cmd /c start "" /wait /high /affinity F powershell -NoProfile -ExecutionPolicy Bypass `
    -File .\experiments\intel-cpu\run_bench.ps1 -Label ptl-u7-355-win-ac-pcores
Remove-Item Env:DOCLING_NUM_THREADS

# A3 (16 cores in this example)
$env:DOCLING_NUM_THREADS = "16"
.\experiments\intel-cpu\run_bench.ps1 -Label ptl-u7-355-win-ac-allcores
Remove-Item Env:DOCLING_NUM_THREADS
```

If PowerShell blocks the script: `Set-ExecutionPolicy -Scope Process Bypass`.

### Quick functional check (5 minutes)

```text
run_bench.sh  --label smoke --docs ocr_test normal_4pages --runs 1
run_bench.ps1 -Label smoke -Docs ocr_test,normal_4pages -Runs 1
```

Delete the `smoke` folder afterwards; do not commit it.

## 5. What to record by hand

Add a `notes.md` in the result folder:

- Laptop model, BIOS version, OEM performance mode, AC / battery.
- Windows power mode or Linux power profile, ambient conditions, lid open.
- Defender exclusion yes / no (Windows).
- Anything unusual (thermal throttling seen in Task Manager / `turbostat`).

Optional on Linux: `sudo turbostat --quiet --interval 5 --show Busy%,Bzy_MHz,PkgWatt,PkgTmp`
in a second terminal, saved to `turbostat.txt`.

## 6. Commit the results

```text
git add experiments/intel-cpu/results/A-baseline/<label>
git commit -m "intel-cpu: A baseline results <label>"
git push origin intel-cpu-exp
```

Never push to `upstream` or to `main`.

## 7. Notes for hybrid CPUs

- OpenMP (torch) splits work statically across threads. A team that spans
  P-cores and E-cores finishes at E-core speed. Compare A2 against A3.
- LP E-cores are slow and outside the main L3; a thread there can slow a
  whole team. Thread Director usually avoids them for foreground work, but
  verify with A2 vs A1.
- OpenVINO (experiments B, C) has `SCHEDULING_CORE_TYPE` (ANY_CORE,
  PCORE_ONLY, ECORE_ONLY) and `ENABLE_HYPER_THREADING`; these are tested in
  B and C, not in A.
- Thread counts larger than the P-core count can still help for throughput
  (several documents) even when they hurt single-document latency.
