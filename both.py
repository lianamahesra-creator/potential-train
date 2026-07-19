#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
  BOTH CONTINUOUS — LLM + Mining simultaneously, never pausing

  Architecture:
    LLM    →  PyTorch process, bitsandbytes NF4 4-bit, capped VRAM
    Mining  →  bash run.sh as separate OS process, own CUDA context
    GPU     →  hardware scheduler shares SMs between both

  No MPS needed. No AWQ/GPTQ. No Triton. Single file.

  Usage:
    python3 both.py
    python3 both.py --llm-vram 6 --max-tokens 128
    python3 both.py --model Qwen/Qwen2.5-3B-Instruct --llm-vram 5
    python3 both.py --miner-script ./my_mining_script.sh
═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import signal
import subprocess
import threading
import json
import queue
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ── Disable Triton / torch compile (prevents C compiler errors) ──
os.environ["TRITON_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
#  CONFIG
# ============================================================

@dataclass
class Config:
    # ── LLM ──
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_max_vram_gb: float = 8.0
    max_new_tokens: int = 256
    temperature: float = 0.8
    llm_pause_between: float = 0.05
    prompts_file: str = ""

    # ── Mining ──
    miner_script: str = "./run.sh"

    # ── System ──
    thermal_limit_c: int = 83
    thermal_resume_c: int = 75
    monitor_interval_s: int = 10
    stats_file: str = "stats.json"
    max_miner_restarts: int = 20
    log_file: str = "both.log"


# ============================================================
#  BUILT-IN PROMPTS
# ============================================================

DEFAULT_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot discovering emotions.",
    "What are the key differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a poem about the ocean at night.",
    "Explain how neural networks learn from data.",
    "What is the significance of prime numbers in cryptography?",
    "Describe the architecture of a modern CPU.",
    "Write a dialogue between two scientists debating AI safety.",
    "Explain blockchain technology to a 10-year-old.",
    "What are the main challenges in space exploration today?",
    "Describe the lifecycle of a star from birth to death.",
    "Write a haiku about machine learning.",
    "Explain the double-slit experiment simply.",
    "What is the Turing test and why does it matter?",
    "Describe how vaccines work at the molecular level.",
    "Write a short tale about time travel paradoxes.",
    "Explain the concept of entropy in thermodynamics.",
    "What makes a good software API design?",
    "Describe the water cycle and its importance to life.",
    "Explain quantum entanglement to a beginner.",
    "Write clear instructions for making a paper airplane.",
    "What is dark matter and why do scientists think it exists?",
    "Describe how GPS satellites determine your location.",
    "Write a limerick about programming bugs.",
    "Explain the CAP theorem in distributed systems.",
    "What causes the northern lights?",
    "Describe the difference between AI, ML, and deep learning.",
    "Write a product description for a fictional smart device.",
    "Explain how public-key cryptography works.",
    "What is the observer effect in quantum mechanics?",
    "Describe the structure of DNA and how it replicates.",
    "Write a persuasive argument for exploring Mars.",
    "How does a neural network differ from a human brain?",
    "Explain the Doppler effect with everyday examples.",
    "What role does mitochondria play in a cell?",
    "Describe the history of the internet in 200 words.",
    "Write a mystery story opening set in a library.",
    "How does machine translation actually work?",
    "Explain supply and demand using a lemonade stand.",
]


# ============================================================
#  LOGGING
# ============================================================

_log_lock = threading.Lock()
_log_file_path = None


def setup_logging(log_file: str):
    global _log_file_path
    _log_file_path = log_file
    with open(log_file, "w") as f:
        f.write(f"=== Both Continuous — {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")


def log(msg: str, level: str = "INFO"):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    with _log_lock:
        print(line, file=sys.stderr, flush=True)
        if _log_file_path:
            try:
                with open(_log_file_path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass


# ============================================================
#  MPS CLEANUP
# ============================================================

def cleanup_mps():
    """Kill leftover MPS daemons from previous crashed runs."""
    for cmd in [
        ["sudo", "killall", "-9", "nvidia-cuda-mps-control"],
        ["sudo", "killall", "-9", "nvidia-cuda-mps-server"],
        ["sudo", "nvidia-smi", "-i", "0", "-c", "DEFAULT"],
        ["sudo", "rm", "-rf", "/tmp/nvidia-mps"],
        ["sudo", "rm", "-rf", "/tmp/nvidia-mps-log"],
    ]:
        try:
            subprocess.run(cmd, capture_output=True, timeout=5)
        except Exception:
            pass


# ============================================================
#  GPU MONITOR
# ============================================================

class GPUMonitor:
    @staticmethod
    def query() -> dict:
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu="
                    "utilization.gpu,"
                    "memory.used,"
                    "memory.total,"
                    "memory.free,"
                    "temperature.gpu,"
                    "power.draw",
                    "--format=csv,nounits,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            p = [x.strip() for x in r.stdout.strip().split(",")]
            return {
                "gpu_util": int(p[0]),
                "mem_used_mb": int(p[1]),
                "mem_total_mb": int(p[2]),
                "mem_free_mb": int(p[3]),
                "temp_c": int(p[4]),
                "power_w": float(p[5]),
            }
        except Exception:
            return {
                "gpu_util": 0,
                "mem_used_mb": 0,
                "mem_total_mb": 24000,
                "mem_free_mb": 0,
                "temp_c": 0,
                "power_w": 0.0,
            }

    @staticmethod
    def get_total_vram_gb() -> float:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return int(r.stdout.strip()) / 1024
        except Exception:
            return 24.0


# ============================================================
#  CUDA CHECK
# ============================================================

def verify_cuda() -> tuple:
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "torch.cuda.is_available() = False"
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        return True, f"{name} ({vram_gb:.1f} GB)"
    except Exception as e:
        return False, str(e)


# ============================================================
#  DEPENDENCY CHECK
# ============================================================

def check_deps() -> list:
    missing = []
    for pkg in ["torch", "transformers", "bitsandbytes", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


# ============================================================
#  MINER CONTROLLER  (runs bash run.sh)
# ============================================================

class ContinuousMiner:
    """Runs the mining script as a separate OS process.
    Gets its own CUDA context — independent from PyTorch."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None
        self.running = False
        self.paused = False
        self.start_time = 0.0
        self.restart_count = 0

    def start(self):
        if self.proc and self.proc.poll() is None:
            return

        cmd = ["bash", self.cfg.miner_script]
        log(f"MINER start: {' '.join(cmd)}")

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            self.running = True
            self.paused = False
            self.start_time = time.time()
            log(f"MINER running  PID={self.proc.pid}")
        except FileNotFoundError:
            log(f"ERROR: script not found: {self.cfg.miner_script}", "ERROR")
            self.running = False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            log("MINER stopping...")
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            except ProcessLookupError:
                pass
        self.proc = None
        self.running = False
        self.paused = False

    def pause(self):
        """Hard-pause via SIGSTOP (thermal protection)."""
        if self.proc and self.proc.poll() is None and not self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGSTOP)
                self.paused = True
                log("MINER paused (thermal)")
            except ProcessLookupError:
                self.running = False

    def resume(self):
        """Resume after thermal cooldown."""
        if self.proc and self.proc.poll() is None and self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGCONT)
                self.paused = False
                log("MINER resumed")
            except ProcessLookupError:
                self.running = False

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def get_uptime(self) -> float:
        if self.running and self.start_time > 0:
            return time.time() - self.start_time
        return 0.0


# ============================================================
#  LLM CONTROLLER  (bitsandbytes NF4 4-bit)
# ============================================================

class ContinuousLLM:
    """Loads a HuggingFace model with bitsandbytes NF4 quantization.
    Runs generation in a non-stop loop in a background thread."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.running = False
        self.total_tokens = 0
        self.total_generations = 0
        self.total_time = 0.0
        self.current_tok_per_sec = 0.0
        self.last_prompt = ""
        self.last_response = ""
        self._prompts = list(DEFAULT_PROMPTS)

    def load(self):
        import torch

        total_gb = GPUMonitor.get_total_vram_gb()
        fraction = self.cfg.llm_max_vram_gb / total_gb
        fraction = min(fraction, 0.90)

        log(f"LLM loading: {self.cfg.model_path}")
        log(f"  GPU total:      {total_gb:.1f} GB")
        log(f"  LLM VRAM cap:   {self.cfg.llm_max_vram_gb:.1f} GB ({fraction:.0%})")
        log(f"  Mining budget:  ~{max(total_gb - self.cfg.llm_max_vram_gb - 1.0, 0):.1f} GB")
        log(f"  Quantization:   bitsandbytes NF4 (4-bit)")

        # Cap LLM VRAM before any CUDA work
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.cuda.empty_cache()

        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        actual_gb = torch.cuda.memory_allocated() / 1024**3
        log(f"LLM loaded. VRAM used: {actual_gb:.2f} GB")

        # Load custom prompts if provided
        if self.cfg.prompts_file and os.path.exists(self.cfg.prompts_file):
            with open(self.cfg.prompts_file, "r") as f:
                custom = [line.strip() for line in f if line.strip()]
            if custom:
                self._prompts = custom
                log(f"  Prompts: {len(custom)} from {self.cfg.prompts_file}")
            else:
                log(f"  Prompts file empty, using {len(self._prompts)} defaults")
        else:
            log(f"  Prompts: {len(self._prompts)} built-in")

    def generate_once(self, prompt: str) -> dict:
        import torch

        t0 = time.time()

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        elapsed = time.time() - t0
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids.shape[1] - input_len

        response = self.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        )

        del inputs, output_ids
        torch.cuda.empty_cache()

        return {
            "response": response,
            "tokens": new_tokens,
            "elapsed_s": round(elapsed, 3),
            "tok_per_sec": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
            "prompt": prompt[:100],
        }

    def run_continuous(
        self,
        result_queue: queue.Queue,
        shutdown_event: threading.Event,
    ):
        """Non-stop generation loop."""
        import torch

        self.running = True
        cycle = 0

        log("LLM continuous loop STARTED")

        while not shutdown_event.is_set():
            try:
                prompt = self._prompts[cycle % len(self._prompts)]
                cycle += 1

                result = self.generate_once(prompt)

                self.total_tokens += result["tokens"]
                self.total_generations += 1
                self.total_time += result["elapsed_s"]
                self.current_tok_per_sec = result["tok_per_sec"]
                self.last_prompt = result["prompt"]
                self.last_response = result["response"][:300]

                if result_queue.full():
                    try:
                        result_queue.get_nowait()
                    except queue.Empty:
                        pass
                result_queue.put(result)

                if self.cfg.llm_pause_between > 0:
                    time.sleep(self.cfg.llm_pause_between)

            except torch.cuda.OutOfMemoryError:
                log("LLM OOM — clearing cache, waiting 15s", "WARN")
                torch.cuda.empty_cache()
                time.sleep(15)
            except Exception as e:
                log(f"LLM error: {type(e).__name__}: {e}", "ERROR")
                time.sleep(5)

        self.running = False
        log("LLM continuous loop STOPPED")


# ============================================================
#  THERMAL GUARD
# ============================================================

class ThermalGuard:
    """Monitors GPU temperature. Pauses mining if too hot."""

    def __init__(
        self,
        miner: ContinuousMiner,
        cfg: Config,
        shutdown_event: threading.Event,
    ):
        self.miner = miner
        self.cfg = cfg
        self._shutdown = shutdown_event
        self.thermal_pauses = 0

    def run(self):
        log("Thermal guard started")
        while not self._shutdown.is_set():
            time.sleep(5)

            if not self.miner.running:
                continue

            temp = GPUMonitor.query().get("temp_c", 0)

            if temp >= self.cfg.thermal_limit_c and not self.miner.paused:
                self.thermal_pauses += 1
                log(
                    f"THERMAL: {temp}°C >= {self.cfg.thermal_limit_c}°C "
                    f"— pausing mining (#{self.thermal_pauses})",
                    "WARN",
                )
                self.miner.pause()

            if self.miner.paused and temp <= self.cfg.thermal_resume_c:
                log(f"THERMAL: {temp}°C <= {self.cfg.thermal_resume_c}°C — resuming")
                self.miner.resume()

        if self.miner.paused:
            self.miner.resume()

        log("Thermal guard stopped")


# ============================================================
#  STATS DISPLAY
# ============================================================

class StatsDisplay:
    """Live terminal dashboard + JSON stats file."""

    def __init__(
        self,
        miner: ContinuousMiner,
        llm: ContinuousLLM,
        thermal: ThermalGuard,
        cfg: Config,
    ):
        self.miner = miner
        self.llm = llm
        self.thermal = thermal
        self.cfg = cfg

    def run(
        self,
        result_queue: queue.Queue,
        shutdown_event: threading.Event,
    ):
        time.sleep(8)

        w = 92
        print()
        print("=" * w)
        print(
            f" {'TIME':<10}│ {'GPU%':<6}│ {'VRAM':<16}│ "
            f"{'TEMP':<6}│ {'WATTS':<7}│ "
            f"{'tok/s':<8}│ {'GENS':<6}│ {'MINER':<9}"
        )
        print("─" * w)

        while not shutdown_event.is_set():
            time.sleep(self.cfg.monitor_interval_s)

            g = GPUMonitor.query()
            ts = time.strftime("%H:%M:%S")
            mem = f"{g['mem_used_mb']}/{g['mem_total_mb']}MB"
            tps = f"{self.llm.current_tok_per_sec:.1f}" if self.llm.running else "..."
            gens = self.llm.total_generations

            if self.miner.paused:
                m = "PAUSED"
            elif self.miner.is_alive():
                m = "RUNNING"
            else:
                m = "DEAD"

            temp = g["temp_c"]
            pwr = g["power_w"]

            line = (
                f"{ts:<10}│ {g['gpu_util']}%{'':<3}│ {mem:<16}│ "
                f"{temp}°C{'':<2}│ {pwr:.0f}W{'':<3}│ "
                f"{tps:<8}│ {gens:<6}│ {m}"
            )
            print(line)

            # Write stats JSON
            try:
                avg_tps = (
                    round(self.llm.total_tokens / max(self.llm.total_time, 0.01), 1)
                    if self.llm.total_time > 0
                    else 0
                )
                data = {
                    "timestamp": ts,
                    "gpu": {
                        "utilization_pct": g["gpu_util"],
                        "vram_used_mb": g["mem_used_mb"],
                        "vram_total_mb": g["mem_total_mb"],
                        "temperature_c": temp,
                        "power_watts": pwr,
                    },
                    "llm": {
                        "running": self.llm.running,
                        "tok_per_sec": self.llm.current_tok_per_sec,
                        "avg_tok_per_sec": avg_tps,
                        "total_generations": self.llm.total_generations,
                        "total_tokens": self.llm.total_tokens,
                        "last_prompt": self.llm.last_prompt[:80],
                        "last_response": self.llm.last_response[:150],
                    },
                    "miner": {
                        "running": self.miner.is_alive(),
                        "paused": self.miner.paused,
                        "uptime_s": round(self.miner.get_uptime()),
                        "restart_count": self.miner.restart_count,
                    },
                    "thermal": {
                        "pause_events": self.thermal.thermal_pauses,
                        "limit_c": self.cfg.thermal_limit_c,
                        "resume_c": self.cfg.thermal_resume_c,
                    },
                }
                with open(self.cfg.stats_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

        self._print_summary(w)

    def _print_summary(self, w: int):
        if self.llm.total_generations == 0:
            return

        avg_tps = self.llm.total_tokens / max(self.llm.total_time, 0.01)
        uptime = self.miner.get_uptime()

        print("=" * w)
        print("  SESSION SUMMARY")
        print(f"    LLM generations:   {self.llm.total_generations}")
        print(f"    Total tokens:      {self.llm.total_tokens}")
        print(f"    Avg speed:         {avg_tps:.1f} tok/s")
        print(f"    Total LLM time:    {self.llm.total_time:.0f}s")
        print(f"    Miner uptime:      {uptime:.0f}s ({uptime/60:.1f} min)")
        print(f"    Miner restarts:    {self.miner.restart_count}")
        print(f"    Thermal pauses:    {self.thermal.thermal_pauses}")
        print("=" * w)


# ============================================================
#  MAIN ORCHESTRATOR
# ============================================================

class BothContinuous:
    """
    Runs LLM and Mining simultaneously, forever.

    ┌──────────────────────────────────────────┐
    │              Main Process                │
    │                                          │
    │  ┌────────────────────┐                  │
    │  │ LLM Thread         │  PyTorch CUDA    │
    │  │ generate() loop    │  context #1      │
    │  │ forever            │  capped at N GB  │
    │  └────────────────────┘                  │
    │                                          │
    ├──────────────────────────────────────────┤
    │              Miner Process               │
    │                                          │
    │  ┌────────────────────┐                  │
    │  │ bash run.sh        │  its own CUDA    │
    │  │ runs forever       │  context #2      │
    │  │                    │  uses remaining  │
    │  └────────────────────┘                  │
    │                                          │
    ├──────────────────────────────────────────┤
    │          GPU Hardware Scheduler          │
    │  Shares SMs between both CUDA contexts   │
    │  Both run simultaneously                 │
    └──────────────────────────────────────────┘
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.miner = ContinuousMiner(cfg)
        self.llm = ContinuousLLM(cfg)
        self.result_queue: queue.Queue = queue.Queue(maxsize=50)
        self._shutdown = threading.Event()
        self._threads: list = []

    def start(self):
        total_gb = GPUMonitor.get_total_vram_gb()
        mining_gb = max(total_gb - self.cfg.llm_max_vram_gb - 1.0, 0)

        print()
        print("╔" + "═" * 60 + "╗")
        print("║      BOTH CONTINUOUS — LLM + Mining, never pausing         ║")
        print("╠" + "═" * 60 + "╣")
        print(f"║      GPU:          {total_gb:.0f} GB total                               ║")
        print(f"║      LLM:          {self.cfg.llm_max_vram_gb:.0f} GB (bitsandbytes 4-bit)              ║")
        print(f"║      Mining:       ~{mining_gb:.0f} GB (remaining VRAM)                   ║")
        print(f"║      Model:        {self.cfg.model_path[:38]:<38}       ║")
        print(f"║      Miner:        {self.cfg.miner_script:<38}       ║")
        print(f"║      Thermal:      {self.cfg.thermal_limit_c}°C limit / {self.cfg.thermal_resume_c}°C resume                    ║")
        print("╚" + "═" * 60 + "╝")
        print()

        # ── Step 0: Cleanup MPS leftovers ──
        log("Cleaning up any leftover MPS state...")
        cleanup_mps()
        time.sleep(1)

        # ── Step 1: Verify CUDA ──
        ok, info = verify_cuda()
        if ok:
            log(f"CUDA OK: {info}")
        else:
            log(f"FATAL: CUDA not working: {info}", "ERROR")
            log("  Try: pip install torch")
            log("  Or:  reboot the machine")
            return

        # ── Step 2: Check deps ──
        missing = check_deps()
        if missing:
            log(f"Missing packages: {', '.join(missing)}", "WARN")
            log(f"  Fix: pip install {' '.join(missing)}")
            return

        # ── Step 3: Load LLM ──
        try:
            self.llm.load()
        except Exception as e:
            log(f"FATAL: Failed to load model: {e}", "ERROR")
            log("  Try: pip install bitsandbytes accelerate")
            log("  Or:  --model Qwen/Qwen2.5-3B-Instruct --llm-vram 5")
            return

        free_mb = GPUMonitor.query().get("mem_free_mb", 0)
        log(f"Free VRAM after LLM load: {free_mb} MB")

        if free_mb < 512:
            log(
                f"WARNING: Only {free_mb}MB free. "
                f"Try: --llm-vram {self.cfg.llm_max_vram_gb - 2}",
                "WARN",
            )

        # ── Step 4: Start miner ──
        self.miner.start()
        time.sleep(3)

        if not self.miner.is_alive():
            log("WARNING: Miner failed to start. Continuing with LLM only.", "WARN")
            log(f"  Script: {self.cfg.miner_script}")
            log(f"  Exists: {os.path.exists(self.cfg.miner_script)}")
            log("  Check:  bash run.sh  (test it manually first)")
        else:
            time.sleep(2)

        # ── Step 5: Create thermal guard ──
        thermal = ThermalGuard(self.miner, self.cfg, self._shutdown)

        # ── Step 6: Create stats display ──
        stats = StatsDisplay(self.miner, self.llm, thermal, self.cfg)

        # ── Step 7: Start LLM continuous loop ──
        llm_thread = threading.Thread(
            target=self.llm.run_continuous,
            args=(self.result_queue, self._shutdown),
            daemon=True,
            name="llm-loop",
        )
        llm_thread.start()
        self._threads.append(llm_thread)

        # ── Step 8: Start thermal guard ──
        thermal_thread = threading.Thread(
            target=thermal.run,
            daemon=True,
            name="thermal",
        )
        thermal_thread.start()
        self._threads.append(thermal_thread)

        # ── Step 9: Start stats display ──
        stats_thread = threading.Thread(
            target=stats.run,
            args=(self.result_queue, self._shutdown),
            daemon=True,
            name="stats",
        )
        stats_thread.start()
        self._threads.append(stats_thread)

        log("All systems running. Press Ctrl+C to stop.\n")

        # ── Step 10: Main loop — keep alive, auto-restart miner ──
        try:
            while not self._shutdown.is_set():
                time.sleep(3)

                if self.miner.running and not self.miner.is_alive():
                    self.miner.restart_count += 1
                    if self.miner.restart_count > self.cfg.max_miner_restarts:
                        log(
                            f"Miner crashed {self.miner.restart_count} times. "
                            "Giving up. LLM continues alone.",
                            "WARN",
                        )
                        self.miner.running = False
                        continue
                    log(
                        f"Miner died — restart #{self.miner.restart_count} in 5s...",
                        "WARN",
                    )
                    time.sleep(5)
                    self.miner.start()

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        log("Shutting down...")
        self._shutdown.set()
        self.miner.stop()
        for t in self._threads:
            t.join(timeout=5)
        log("Shutdown complete.")


# ============================================================
#  ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM + Mining simultaneously on the same GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Default: runs run.sh with Qwen2.5-7B
  python3 both.py

  # Smaller model, more room for mining
  python3 both.py --model Qwen/Qwen2.5-3B-Instruct --llm-vram 5

  # Faster, shorter generations
  python3 both.py --max-tokens 128 --pause 0.01

  # Custom prompts
  python3 both.py --prompts-file my_prompts.txt

  # Custom miner script
  python3 both.py --miner-script ./my_miner.sh
        """,
    )

    # ── LLM ──
    llm_g = parser.add_argument_group("LLM options")
    llm_g.add_argument(
        "--model", type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    llm_g.add_argument(
        "--llm-vram", type=float, default=8.0,
        help="Max GB VRAM for LLM (default: 8)",
    )
    llm_g.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens per generation (default: 256)",
    )
    llm_g.add_argument(
        "--temperature", type=float, default=0.8,
        help="Generation temperature (default: 0.8)",
    )
    llm_g.add_argument(
        "--pause", type=float, default=0.05,
        help="Seconds between generations (default: 0.05)",
    )
    llm_g.add_argument(
        "--prompts-file", type=str, default="",
        help="Text file with one prompt per line",
    )

    # ── Mining ──
    mine_g = parser.add_argument_group("Mining options")
    mine_g.add_argument(
        "--miner-script", type=str, default="./run.sh",
        help="Path to mining bash script (default: ./run.sh)",
    )

    # ── System ──
    sys_g = parser.add_argument_group("System options")
    sys_g.add_argument(
        "--thermal", type=int, default=83,
        help="GPU thermal limit °C (default: 83)",
    )
    sys_g.add_argument(
        "--thermal-resume", type=int, default=75,
        help="Resume mining at °C (default: 75)",
    )
    sys_g.add_argument(
        "--stats-file", type=str, default="stats.json",
        help="Stats JSON output (default: stats.json)",
    )
    sys_g.add_argument(
        "--log-file", type=str, default="both.log",
        help="Log file (default: both.log)",
    )

    args = parser.parse_args()

    # ── Validate miner script ──
    if not os.path.exists(args.miner_script):
        print(f"ERROR: Miner script not found: {args.miner_script}")
        print()
        print("Create run.sh with your mining commands:")
        print()
        print("  #!/bin/bash")
        print("  ./forge \\")
        print("    --algorithm pearlhash \\")
        print("    --pool prl.kryptex.network:7048 \\")
        print("    --wallet YOUR_WALLET_HERE")
        print()
        print("  chmod +x run.sh")
        print()
        print("Or specify: --miner-script /path/to/script.sh")
        sys.exit(1)

    # ── Setup ──
    setup_logging(args.log_file)

    cfg = Config(
        model_path=args.model,
        llm_max_vram_gb=args.llm_vram,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        llm_pause_between=args.pause,
        prompts_file=args.prompts_file,
        miner_script=args.miner_script,
        thermal_limit_c=args.thermal,
        thermal_resume_c=args.thermal_resume,
        stats_file=args.stats_file,
        log_file=args.log_file,
    )

    # ── Run ──
    app = BothContinuous(cfg)

    def sig_handler(signum, frame):
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    app.start()


if __name__ == "__main__":
    main()
