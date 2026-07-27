"""
run_benchmark.py
Benchmark unificado -- BitNet b1.58 (2B-4T) HLS Simulation
Hardware: Intel Core i5-7400 (Kaby Lake)

VERSAO 4.0 - UNIFICADO (HLS_min_dist_unified)
===============================================
- Adaptado para o diretorio unificado: codigo e dados (.bin) estao
  na mesma pasta. Nao ha mais SHARED_DIR separado.
- Seleciona versao via argumento --version (0, 1, 2) e passa -DBITNET_VERSION
  na compilacao.
- Usa um unico grupo de 5 eventos perf stat (sem multiplexacao).
  O i5-7400 tem 4 contadores programaveis + 2 fixos (cycles, instructions).
  Grupo: cycles (fixo), instructions (fixo), branches, branch-misses,
         L1-dcache-loads (3 dos 4 programaveis -- zero multiplexacao).
- Analise de cache vem exclusivamente do perf mem (PEBS).
- Mantém todos os cenarios, compilacoes e relatorio do formato original.
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from logging import getLogger
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

# ============================================================================
# Estrutura de pastas — igual ao dump_activations.py / timing_breakdown.py
# ============================================================================
BASE_DIR   = Path(os.getcwd()).resolve()
# No diretório unificado, os .bin ficam na mesma pasta do código.
# SHARED_DIR aponta para BASE_DIR por compatibilidade com o restante do script.
SHARED_DIR = BASE_DIR
VERSION    = "HLS_min_dist_unified"

# BITNET_VERSION é controlado pelo argumento --version em runtime.
# WEIGHTS_FILE é definido após parse de argumentos (veja main()).
WEIGHTS_FILE = "weights.bin"  # default; substituído em main()

CPU_LABEL = "Intel Core i5-7400 (Kaby Lake)"


# ============================================================================
# Tokenizador (inlined de tokenizer.py)
# ============================================================================

try:
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
except ImportError:
    print("Erro: a biblioteca 'tiktoken' não está instalada.")
    print("Execute:  pip install tiktoken")
    sys.exit(1)

logger = getLogger(__name__)

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]


class Tokenizer:
    """Tokenizador BPE (Tiktoken / Llama-3) embutido."""

    special_tokens: Dict[str, int]
    num_reserved_special_tokens = 256
    pat_str = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    def __init__(self, model_path: str):
        assert os.path.isfile(model_path), f"tokenizer.model não encontrado: {model_path}"
        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.n_words: int = self.model.n_vocab
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.pad_id: int = self.n_words - 1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        assert type(s) is str
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000
        MAX_NO_WHITESPACES_CHARS = 25_000
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0
        for i in range(len(s)):
            is_now_space = s[i].isspace()
            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


# ============================================================================
# Tokens
# ============================================================================

SHORT_PROMPT_TEXT   = "What is the main purpose of neural networks?"
SHORT_PROMPT_TOKENS = [128000, 3923, 374, 279, 1925, 7580, 315, 30828, 14488, 30]

LONG_PROMPT_TEXT = (
    "Can you explain the main differences between standard floating-point "
    "models and ternary models like BitNet b1.58, specifically focusing on "
    "how activation quantization and ternary weights reduce memory "
    "bandwidth requirements and computational latency during hardware "
    "simulation on FPGA accelerators?"
)
LONG_PROMPT_TOKENS = [128000, 6854, 499, 10552, 279, 1925, 12062, 1990, 5410, 19596,
                      16983, 4211, 323, 72717, 661, 4211, 1093, 6631, 7099, 293, 16,
                      13, 2970, 11, 11951, 21760, 389, 1268, 15449, 10484, 2065, 323,
                      72717, 661, 14661, 8108, 5044, 34494, 8670, 323, 55580, 40370,
                      2391, 12035, 19576, 389, 90562, 14511, 3046, 30]

# ============================================================================
# Cenários
# ============================================================================

SCENARIO_SPECS = [
    {
        "id": "short_short",
        "name": "Short Prompt, Short Gen",
        "focus": "Decode pesado, prefill leve",
        "prompt_text": SHORT_PROMPT_TEXT,
        "prompt_tokens": SHORT_PROMPT_TOKENS,
        "gen_len": 10,
    },
    {
        "id": "short_long",
        "name": "Short Prompt, Long Gen",
        "focus": "Decode dominante, cache KV crescendo",
        "prompt_text": SHORT_PROMPT_TEXT,
        "prompt_tokens": SHORT_PROMPT_TOKENS,
        "gen_len": 20,
    },
    {
        "id": "long_short",
        "name": "Long Prompt, Short Gen",
        "focus": "Prefill dominante, decode rapido",
        "prompt_text": LONG_PROMPT_TEXT,
        "prompt_tokens": LONG_PROMPT_TOKENS,
        "gen_len": 10,
    },
    {
        "id": "long_long",
        "name": "Long Prompt, Long Gen",
        "focus": "Prefill longo + decode pesado, pressao maxima de memoria",
        "prompt_text": LONG_PROMPT_TEXT,
        "prompt_tokens": LONG_PROMPT_TOKENS,
        "gen_len": 20,
    },
]

# ============================================================================
# Flags de compilação
# ============================================================================

COMMON_FLAGS = ["-g", "-Wall", "-fno-inline", "-fno-inline-functions", "-fopt-info-vec"]

COMPILATIONS = [
    {"name": "O0",
     "flags": ["-O0"] + COMMON_FLAGS},
    {"name": "O2_baseline",
     "flags": ["-O2", "-fno-tree-vectorize", "-fno-unroll-loops"] + COMMON_FLAGS},
    {"name": "SIMD",
     "flags": ["-O2", "-mavx2", "-ftree-vectorize", "-fno-unroll-loops"] + COMMON_FLAGS},
    {"name": "Unroll",
     "flags": ["-O2", "-fno-tree-vectorize", "-funroll-loops"] + COMMON_FLAGS},
    {"name": "Accumulated",
     "flags": ["-O2", "-mavx2", "-ftree-vectorize", "-funroll-loops"] + COMMON_FLAGS},
    {"name": "O3_reference",
     "flags": ["-O3"] + COMMON_FLAGS},
]

# ============================================================================
# Eventos de PMU -- ESTRATEGIA SEM MULTIPLEXACAO
#
# O i5-7400 tem 4 contadores programaveis + 2 fixos (cycles, instructions).
# Grupo unico de 6 eventos:
#   cycles       (fixo)
#   instructions (fixo)
#   branches     (programavel 1/4)
#   branch-misses(programavel 2/4)
#   L1-dcache-loads (programavel 3/4)
#   cpu-clock    (SOFTWARE -- nao consome contador de hardware)
# Resultado: 3 dos 4 programaveis usados, zero multiplexacao.
#
# Analise de cache (L1/L2/L3/RAM) vem exclusivamente do perf mem (PEBS).
# ============================================================================

PERF_EVENTS = [
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "L1-dcache-loads",
    "cpu-clock",       # evento de software: tempo de CPU em ns (calcula avg_clock_ghz)
]

# Latencia ideal de L1 no i5-7400 (ciclos)
L1_IDEAL_LATENCY_CYCLES = 4
CPU_BASE_CLOCK_GHZ      = 3.0


# ============================================================================
# Funções auxiliares
# ============================================================================

def check_perf_permissions():
    """Verifica se o perf está disponível e com permissão suficiente."""
    try:
        res = subprocess.run(["perf", "--version"], capture_output=True, text=True)
        if res.returncode != 0:
            return False, "perf não está instalado ou não é executável."
    except FileNotFoundError:
        return False, "Comando perf não encontrado."

    if os.geteuid() == 0:
        return True, "perf disponível (rodando como root)."

    paranoid_file = "/proc/sys/kernel/perf_event_paranoid"
    if os.path.exists(paranoid_file):
        try:
            with open(paranoid_file, "r") as f:
                level = int(f.read().strip())
                if level > 1:
                    return False, (
                        f"perf_event_paranoid={level} e não é root. "
                        "Para hardware counters, rode com 'sudo' ou execute: "
                        "'sudo sysctl -w kernel.perf_event_paranoid=1'"
                    )
        except Exception as e:
            return False, f"Não foi possível ler {paranoid_file}: {e}"

    return True, "perf disponível."


def check_required_files() -> bool:
    """Verifica se os arquivos necessários existem."""
    required_code = ["bitnet.c", "testbench.c", "bitnet.h", "config.h"]
    required_shared = [
        "tokenizer.model", "embeddings.bin", WEIGHTS_FILE,
        "norms.bin", "scales.bin",
    ]

    missing = []
    for fname in required_code:
        if not os.path.isfile(os.path.join(BASE_DIR, fname)):
            missing.append(f"{fname} (esperado em {BASE_DIR})")
    for fname in required_shared:
        if not os.path.isfile(os.path.join(SHARED_DIR, fname)):
            missing.append(f"{fname} (esperado em {SHARED_DIR})")

    if missing:
        print("\n[ERRO] Arquivos ausentes:")
        for m in missing:
            print(f"  AUSENTE: {m}")
        return False

    print(f"[OK] Todos os arquivos necessários foram encontrados. "
          f"(WEIGHTS_FILE detectado: {WEIGHTS_FILE})")
    return True


def run_compile(flags: list, binary_path: str, comp_name: str, results_dir: str,
                bitnet_version: int = 1) -> bool:
    """Compila bitnet.c e testbench.c com as flags especificadas."""
    cmd = (
        ["gcc"]
        + [f"-DBITNET_VERSION={bitnet_version}"]
        + flags
        + [
            str(BASE_DIR / "bitnet.c"),
            str(BASE_DIR / "testbench.c"),
            "-o", binary_path,
            "-lm",
        ]
    )
    print(f"  Comando: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.stderr:
        safe_comp = comp_name.replace(" ", "_")
        vec_report_path = os.path.join(results_dir, f"vec_report_{safe_comp}.txt")
        with open(vec_report_path, "w", encoding="utf-8") as f:
            f.write(res.stderr)
        vectorized_lines = [l for l in res.stderr.split("\n") if "loop vectorized" in l.lower()]
        print(f"  [opt-info-vec] {len(vectorized_lines)} loop(s) reportado(s) como "
              f"vetorizado(s) (relatório completo em {vec_report_path})")

    if res.returncode != 0:
        print(f"  Compilação falhou!\n  Stderr: {res.stderr}")
        return False
    return True


def parse_sim_output(stdout_str: str, tok: Tokenizer) -> dict:
    """Extrai tokens de saída e timings do stdout do binário C."""
    results = {"output_text": "", "timings": {}}
    out_tokens = []
    lines = stdout_str.split("\n")
    current_section = None

    for line in lines:
        line = line.strip()
        if "[OUT]" in line:
            try:
                token_id = int(line.split("[OUT]")[1])
                out_tokens.append(token_id)
            except ValueError:
                pass
        elif "Tempo total de execucao" in line:
            try:
                val = float(line.split(":")[1].split()[0])
                results["timings"]["Total Inference"] = val
            except Exception:
                pass
        elif line.startswith("[") and line.endswith("]"):
            current_section = line
        elif "Tempo gasto:" in line and current_section:
            try:
                val = float(line.split(":")[1].split()[0])
                results["timings"][current_section] = val
            except Exception:
                pass

    if out_tokens and tok:
        try:
            results["output_text"] = tok.decode(out_tokens)
        except Exception as e:
            results["output_text"] = f"[Erro ao decodificar: {e}]"

    return results


def run_perf_stat_single_group(binary_path: str, prompt_tokens_str: str, gen_len: int,
                               perf_available: bool):
    """
    Executa o binario sob perf stat com o grupo unico de 5 eventos.
    Captura stdout do binario e os contadores ao mesmo tempo (uma execucao so).

    Formato CSV do perf stat -x ,:
      value , unit , event_name , run_pct , enabled_time , running_time , ...
    O campo de evento fica no indice 2 (0-based). O valor pode ser '<not counted>'
    se o contador nao foi suportado -- nesses casos o par e ignorado.

    Retorna (stdout_str, stats_dict).
    """
    env = os.environ.copy()
    env["MAX_GEN"] = str(gen_len)
    env["WEIGHTS_FILE"] = WEIGHTS_FILE

    if not perf_available:
        process = subprocess.run(
            [binary_path] + prompt_tokens_str.split(),
            env=env, cwd=SHARED_DIR, capture_output=True, text=True,
        )
        return process.stdout, {}

    events_str = ",".join(PERF_EVENTS)
    cmd = [
        "perf", "stat", "-e", events_str,
        "-x", ",",
        binary_path
    ] + prompt_tokens_str.split()

    result = subprocess.run(cmd, env=env, cwd=SHARED_DIR,
                            capture_output=True, text=True)

    # perf stat envia os contadores para stderr; stdout do binario vai para stdout
    stdout_str = result.stdout

    stats = {}
    for line in result.stderr.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Formato: value,unit,event[,run%,enabled,running,...]
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value_raw = parts[0].strip()
        event_name = parts[2].strip()
        # Remover sufixo de qualificador (:u, :k, etc.) para chave limpa
        event_key = event_name.split(":")[0].strip()
        if not event_key:
            continue
        try:
            stats[event_key] = int(value_raw)
        except ValueError:
            # '<not counted>' ou outro valor nao numerico -- ignora
            continue

    return stdout_str, stats


def run_perf_combined(binary_path: str, prompt_tokens_str: str, gen_len: int,
                      comp_name: str, scen_id: str, results_dir: str,
                      perf_available: bool):
    """
    Execucao principal. Chama run_perf_stat_single_group que executa o
    binario uma unica vez sob perf stat (capturando stdout e contadores
    simultaneamente). Depois executa perf record para o top 10 de funcoes.
    Retorna: (stdout_str, perf_stats dict, top_functions list)
    """
    env = os.environ.copy()
    env["MAX_GEN"] = str(gen_len)
    env["WEIGHTS_FILE"] = WEIGHTS_FILE

    # --- Execucao sob perf stat (captura stdout + contadores em uma so vez) ---
    stdout_str, raw_stats = run_perf_stat_single_group(
        binary_path, prompt_tokens_str, gen_len, perf_available
    )

    # --- Metricas derivadas ---
    cycles        = raw_stats.get("cycles", 0)
    instructions  = raw_stats.get("instructions", 0)
    branches      = raw_stats.get("branches", 0)
    branch_misses = raw_stats.get("branch-misses", 0)
    l1_loads      = raw_stats.get("L1-dcache-loads", 0)
    # cpu-clock: evento de software que retorna tempo de CPU em nanosegundos.
    # GHz = ciclos / nanosegundos  (cycles/ns == 10^9 cycles/s == GHz)
    cpu_clock_ns  = raw_stats.get("cpu-clock", 0)

    all_stats = dict(raw_stats)

    if cycles > 0:
        all_stats["IPC"] = round(instructions / cycles, 2)
    if branches > 0:
        all_stats["branch-miss-rate (%)"] = round((branch_misses / branches) * 100, 2)
    if l1_loads > 0 and instructions > 0:
        all_stats["IPL"] = round(instructions / l1_loads, 2)
    if cpu_clock_ns > 0 and cycles > 0:
        all_stats["avg_clock_ghz"] = round(cycles / cpu_clock_ns, 4)

    if not perf_available:
        return stdout_str, all_stats, []

    # --- Top 10 funcoes (perf record + perf report) ---
    # Execucao adicional necessaria apenas para sampling de simbolos.
    top_functions = []
    perf_data = os.path.join(BASE_DIR, "perf.data")

    cmd_rec = [
        "perf", "record", "-q", "-e", "cycles", "-o", perf_data,
        binary_path
    ] + prompt_tokens_str.split()

    subprocess.run(cmd_rec, env=env, cwd=SHARED_DIR, capture_output=True)

    if os.path.exists(perf_data):
        cmd_rep = [
            "perf", "report", "-i", perf_data, "--stdio", "-n",
            "--stdio-color", "never",
        ]
        res_rep = subprocess.run(cmd_rep, capture_output=True, text=True)
        for line in res_rep.stdout.split("\n"):
            line = line.strip()
            if ("[.]" in line or "[k]" in line) and "%" in line:
                parts = line.split()
                if len(parts) >= 5:
                    overhead = parts[0]
                    symbol_index = -1
                    for idx, p in enumerate(parts):
                        if p in ("[.]", "[k]"):
                            symbol_index = idx
                            break
                    if symbol_index != -1:
                        symbol = " ".join(parts[symbol_index:])
                        top_functions.append({"overhead": overhead, "symbol": symbol})

        def overhead_value(entry):
            try:
                return float(entry["overhead"].replace("%", "").replace(",", "."))
            except ValueError:
                return 0.0

        top_functions.sort(key=overhead_value, reverse=True)
        top_functions = top_functions[:10]

        # --- perf annotate bitlinear ---
        safe_comp = comp_name.replace(" ", "_")
        annotate_file = os.path.join(
            results_dir, f"bitlinear_assembly_{safe_comp}_{scen_id}.txt"
        )
        cmd_ann = ["perf", "annotate", "bitlinear", "-i", perf_data, "--stdio"]
        res_ann = subprocess.run(cmd_ann, capture_output=True, text=True)
        if res_ann.stdout:
            with open(annotate_file, "w", encoding="utf-8") as f:
                f.write(res_ann.stdout)

        try:
            os.remove(perf_data)
        except OSError:
            pass

    return stdout_str, all_stats, top_functions


# Mapeamento de palavras-chave do perf mem para nivel de cache
_MEM_LEVEL_MAP = [
    ("L1",   ["L1"]),
    ("L2",   ["L2"]),
    ("L3",   ["L3", "LLC"]),
    ("RAM",  ["RAM", "DRAM", "Local RAM", "Remote RAM"]),
]


def _classify_level(level_desc: str) -> str:
    """Classifica a descricao de nivel de memoria em L1/L2/L3/RAM/Outro."""
    for label, keywords in _MEM_LEVEL_MAP:
        for kw in keywords:
            if kw.upper() in level_desc.upper():
                return label
    return "Outro"


def build_memory_analysis(mem_sections: List[dict], top_functions: List[dict]) -> dict:
    """
    Constroi dicionario de analise consolidada de memoria a partir dos
    dados do perf mem e do perf report.

    Campos retornados:
      level_distribution  -- dict {L1, L2, L3, RAM, Outro} -> % de loads
      avg_latency_cycles  -- latencia media dos loads em ciclos
      avg_latency_ns      -- latencia media em nanosegundos
      mem_wait_fraction   -- (lat_media - L1_ideal) / lat_media
      bitlinear_overhead_pct -- % de CPU consumida pela funcao bitlinear
      time_lost_mem_pct   -- bitlinear_overhead_pct * mem_wait_fraction
      time_effective_compute_pct -- bitlinear_overhead_pct * (1 - mem_wait_fraction)
    """
    analysis: dict = {
        "level_distribution": {},
        "avg_latency_cycles": None,
        "avg_latency_ns": None,
        "mem_wait_fraction": None,
        "bitlinear_overhead_pct": None,
        "time_lost_mem_pct": None,
        "time_effective_compute_pct": None,
    }

    # Encontrar secao de loads
    load_sec = next((s for s in mem_sections if s["kind"] == "loads"), None)
    if load_sec is not None:
        # Distribuicao por nivel
        bucket: Dict[str, float] = {"L1": 0.0, "L2": 0.0, "L3": 0.0, "RAM": 0.0, "Outro": 0.0}
        for lvl in load_sec.get("levels", []):
            label = _classify_level(lvl["level"])
            bucket[label] = round(bucket.get(label, 0.0) + lvl["overhead_pct"], 2)
        analysis["level_distribution"] = bucket

        # Latencia media
        avg_cyc = load_sec.get("avg_latency_cycles")
        if avg_cyc is not None:
            analysis["avg_latency_cycles"] = avg_cyc
            analysis["avg_latency_ns"] = round(avg_cyc / CPU_BASE_CLOCK_GHZ, 3)
            if avg_cyc > 0:
                frac = (avg_cyc - L1_IDEAL_LATENCY_CYCLES) / avg_cyc
                analysis["mem_wait_fraction"] = round(max(frac, 0.0), 4)

    # Percentagem de CPU da bitlinear (extraida do top 10)
    bitlinear_pct = 0.0
    for fn in top_functions:
        sym = fn.get("symbol", "")
        if "bitlinear" in sym.lower():
            try:
                bitlinear_pct += float(
                    fn["overhead"].replace("%", "").replace(",", ".")
                )
            except ValueError:
                pass
    if bitlinear_pct > 0:
        analysis["bitlinear_overhead_pct"] = round(bitlinear_pct, 2)
        frac = analysis.get("mem_wait_fraction")
        if frac is not None:
            analysis["time_lost_mem_pct"] = round(bitlinear_pct * frac, 2)
            analysis["time_effective_compute_pct"] = round(
                bitlinear_pct * (1.0 - frac), 2
            )

    return analysis


def run_perf_mem_latency(binary_path: str, prompt_tokens_str: str, gen_len: int,
                         comp_name: str, scen_id: str, results_dir: str,
                         perf_available: bool):
    """
    Execucao adicional do binario sob `perf mem record` (PEBS).
    Retorna (mem_sections, raw_report_text).
    """
    if not perf_available:
        return [], ""

    env = os.environ.copy()
    env["MAX_GEN"] = str(gen_len)
    env["WEIGHTS_FILE"] = WEIGHTS_FILE

    perf_mem_data = os.path.join(BASE_DIR, "perf_mem.data")
    cmd_rec = ["perf", "mem", "record", "--ldlat", "3", "-o", perf_mem_data, binary_path] + prompt_tokens_str.split()
    proc = subprocess.run(cmd_rec, env=env, cwd=SHARED_DIR, capture_output=True, text=True)

    if not os.path.exists(perf_mem_data):
        print(f"    AVISO: perf mem record falhou. Stderr: {proc.stderr[:300]}")
        return [], ""

    cmd_rep = [
        "perf", "mem", "report", "-i", perf_mem_data, "--stdio", "-n",
        "--sort=mem",
    ]
    res_rep = subprocess.run(cmd_rep, capture_output=True, text=True)
    raw_report_text = res_rep.stdout

    safe_comp = comp_name.replace(" ", "_")
    raw_path = os.path.join(results_dir, f"mem_latency_raw_{safe_comp}_{scen_id}.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_report_text)

    mem_sections = []
    current_section = None

    for line in raw_report_text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("# Samples:") and "of event" in stripped:
            if current_section is not None:
                mem_sections.append(current_section)
            event_name = stripped.split("of event")[-1].strip().strip("'")
            kind = "loads" if "mem-loads" in event_name else (
                "stores" if "mem-stores" in event_name else "unknown"
            )
            current_section = {
                "event": event_name,
                "kind": kind,
                "total_weight_cycles": None,
                "levels": [],
            }
            continue

        if current_section is None:
            continue

        if stripped.startswith("# Total weight"):
            try:
                current_section["total_weight_cycles"] = int(
                    stripped.split(":")[1].strip()
                )
            except (ValueError, IndexError):
                pass
            continue

        if "%" in stripped and not stripped.startswith("#"):
            parts = stripped.split(None, 2)
            if len(parts) == 3:
                pct_raw, samples_raw, level_desc = parts
                try:
                    pct = float(pct_raw.replace("%", "").replace(",", "."))
                    samples = int(samples_raw)
                except ValueError:
                    continue
                current_section["levels"].append({
                    "level": level_desc.strip(),
                    "overhead_pct": pct,
                    "samples": samples,
                })

    if current_section is not None:
        mem_sections.append(current_section)

    for sec in mem_sections:
        total_samples = sum(lvl["samples"] for lvl in sec["levels"])
        sec["total_samples"] = total_samples
        if sec["total_weight_cycles"] is not None and total_samples > 0:
            avg_cycles = sec["total_weight_cycles"] / total_samples
            sec["avg_latency_cycles"] = round(avg_cycles, 2)
            sec["avg_latency_ns"] = round(avg_cycles / CPU_BASE_CLOCK_GHZ, 3)

    try:
        os.remove(perf_mem_data)
    except OSError:
        pass

    return mem_sections, raw_report_text


def restore_ownership(filepath: str):
    """Restaura ownership ao usuário original quando rodando com sudo."""
    sudo_uid = os.environ.get("SUDO_UID")
    sudo_gid = os.environ.get("SUDO_GID")
    if sudo_uid and sudo_gid:
        try:
            os.chown(filepath, int(sudo_uid), int(sudo_gid))
        except Exception as e:
            print(f"Aviso: não foi possível alterar ownership de {filepath}: {e}")


# ============================================================================
# Logger (Tee)
# ============================================================================

class _Tee:
    """Redireciona sys.stdout para o terminal E para um arquivo simultaneamente."""
    def __init__(self, filepath: str):
        self._terminal = sys.stdout
        self._log = open(filepath, "w", encoding="utf-8")

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()
        sys.stdout = self._terminal


# ============================================================================
# Main
# ============================================================================

def main():
    global WEIGHTS_FILE

    parser = argparse.ArgumentParser(
        description="BitNet b1.58 -- Benchmark Unificado (Intel i5-7400)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execução rápida: 1 cenário × 1 compilação (debug -O0)",
    )
    parser.add_argument(
        "--version",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="Versão do BitNet a compilar (BITNET_VERSION). Padrão: 1",
    )
    args = parser.parse_args()

    # Ajusta arquivo de pesos conforme versão
    WEIGHTS_FILE = "weights_packed.bin" if args.version == 2 else "weights.bin"

    print("==================================================")
    print(f" BitNet b1.58 -- Benchmark Unificado ({VERSION})")
    print("==================================================")
    print(f"Diretorio base:   {BASE_DIR}")
    print(f"Dados (.bin):     {SHARED_DIR}")
    print(f"CPU:              {CPU_LABEL}")
    print(f"BITNET_VERSION:   {args.version}")
    print(f"WEIGHTS_FILE:     {WEIGHTS_FILE}")
    print("\nEstrategia perf stat (grupo unico, sem multiplexacao):")
    print(f"  Eventos: {', '.join(PERF_EVENTS)}")
    print(f"  Contadores fixos: cycles, instructions")
    print(f"  Contadores programaveis usados: 3 de 4")
    print(f"  Analise de cache: exclusivamente via perf mem (PEBS)\n")

    if not check_required_files():
        sys.exit(1)

    perf_available, perf_msg = check_perf_permissions()
    if not perf_available:
        print(f"\n[ERRO FATAL] {perf_msg}")
        print("O benchmark requer perf com permissões de hardware counters.")
        print("Abortando para evitar execução sem dados de performance.\n")
        sys.exit(1)
    print(f"Status do perf: {perf_msg}\n")

    binary_path = str(BASE_DIR / "bitnet_sim_bench")
    results_dir = str(BASE_DIR / f"benchmark_results_v{args.version}")
    os.makedirs(results_dir, exist_ok=True)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"run_{run_ts}.log")
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"Log desta execução: {log_path}\n")

    model_path = BASE_DIR / "tokenizer.model"
    tok = Tokenizer(str(model_path))
    print("Tokenizador carregado com sucesso.")

    scenarios_to_run    = SCENARIO_SPECS
    compilations_to_run = COMPILATIONS

    if args.dry_run:
        print("\n*** MODO DRY-RUN: 1 compilação × 1 cenário ***")
        compilations_to_run = COMPILATIONS[:1]
        scenarios_to_run    = scenarios_to_run[:1]

    all_runs = []
    total_runs = len(compilations_to_run) * len(scenarios_to_run)
    run_idx = 0

    for comp in compilations_to_run:
        print(f"\n{'='*50}")
        print(f" Compilando: {comp['name']}")
        print(f"{'='*50}")

        if not run_compile(comp["flags"], binary_path, comp["name"], results_dir,
                           bitnet_version=args.version):
            print(f"  AVISO: Pulando '{comp['name']}' por erro de compilacao.")
            continue

        print("  OK: Compilacao concluida.")

        for sc in scenarios_to_run:
            run_idx += 1
            print(
                f"\n[{run_idx}/{total_runs}] Cenário: '{sc['name']}' ({sc['id']})"
                f"  (Prompt: {len(sc['prompt_tokens'])} tokens | Geração: {sc['gen_len']} tokens)"
            )

            prompt_tokens_str = " ".join(str(tid) for tid in sc["prompt_tokens"])
            print(f"  Tokens de prompt: {len(sc['prompt_tokens'])} (exato)")

            stdout_str, raw_perf_stats, top_fns = run_perf_combined(
                binary_path, prompt_tokens_str, sc["gen_len"],
                comp["name"], sc["id"], results_dir, perf_available,
            )

            # --- Diagnostico: linhas de carregamento de dados ---
            print("  [Carregamento de dados]")
            load_lines_found = False
            for line in stdout_str.split("\n"):
                if any(kw in line for kw in (
                    "Carregando", "carregados", "nao encontrado", "dummy", "sinteticos"
                )):
                    print(f"    {line.strip()}")
                    load_lines_found = True
            if not load_lines_found:
                print("    (nenhuma linha de carregamento capturada)")

            sim_metrics = parse_sim_output(stdout_str, tok)
            print(f"  Saida decodificada: \"{sim_metrics['output_text']}\"")

            print("\n  [Perf -- Contadores (grupo unico, SEM multiplexacao)]")
            for k, v in raw_perf_stats.items():
                if isinstance(v, int):
                    print(f"    {k:<35}: {v:,}")
                elif isinstance(v, float):
                    print(f"    {k:<35}: {v:.2f}")

            print("\n  [Perf Report - Top 10 Funcoes por Tempo de CPU]")
            for fn in top_fns:
                print(f"    {fn['overhead']} | {fn['symbol']}")

            # --- Latencia por nivel de memoria (PEBS) ---
            print("\n  [Perf Mem - Latencia de loads/stores por nivel (PEBS)]")
            mem_sections, _ = run_perf_mem_latency(
                binary_path, prompt_tokens_str, sc["gen_len"],
                comp["name"], sc["id"], results_dir, perf_available,
            )
            if mem_sections:
                for sec in mem_sections:
                    print(f"    [{sec['kind'].upper()}] latencia media: "
                          f"{sec.get('avg_latency_ns', '?')} ns "
                          f"({sec.get('avg_latency_cycles', '?')} ciclos) "
                          f"| amostras: {sec['total_samples']:,}")
                    for lvl in sec["levels"]:
                        print(f"      {lvl['level']:<14} | {lvl['overhead_pct']:>6.2f}% "
                              f"| {lvl['samples']:,} amostras")
                    if sec["kind"] == "stores" and sec.get("avg_latency_cycles", 0) <= 1.5:
                        print("      AVISO: Latencia de stores ~1 ciclo -- "
                              "evento sem peso de latencia real neste hardware.")
            else:
                print("    (sem dados -- verificar suporte a PEBS/perf mem)")

            # --- Analise consolidada de memoria ---
            memory_analysis = build_memory_analysis(mem_sections, top_fns)
            print("\n  [Analise de Memoria -- Impacto na BitLinear]")
            dist = memory_analysis.get("level_distribution", {})
            if dist:
                print("    Distribuicao de loads:")
                for lv, pct in dist.items():
                    print(f"      {lv:<6}: {pct:>6.2f}%")
            if memory_analysis["avg_latency_cycles"] is not None:
                print(f"    Latencia media loads : {memory_analysis['avg_latency_cycles']} ciclos "
                      f"/ {memory_analysis['avg_latency_ns']} ns")
            if memory_analysis["mem_wait_fraction"] is not None:
                print(f"    Fracao de espera mem : {memory_analysis['mem_wait_fraction']:.4f} "
                      f"(L1 ideal = {L1_IDEAL_LATENCY_CYCLES} ciclos)")
            if memory_analysis["bitlinear_overhead_pct"] is not None:
                print(f"    Overhead bitlinear   : {memory_analysis['bitlinear_overhead_pct']:.2f}% de CPU")
                print(f"    Tempo perdido c/ mem : {memory_analysis['time_lost_mem_pct']:.2f}% de CPU total")
                print(f"    Tempo efetivo comput : {memory_analysis['time_effective_compute_pct']:.2f}% de CPU total")

            print("\n" + "-" * 50)

            run_data = {
                "compilation": comp["name"],
                "compilation_flags": " ".join(comp["flags"]),
                "scenario": sc["name"],
                "scenario_id": sc["id"],
                "scenario_focus": sc.get("focus", ""),
                "prompt_len": len(sc["prompt_tokens"]),
                "gen_len": sc["gen_len"],
                "prompt_text": sc["prompt_text"],
                "prompt_tokens": sc["prompt_tokens"],
                "output_text": sim_metrics["output_text"],
                "c_timings": sim_metrics["timings"],
                "perf_stats": raw_perf_stats,
                "top_functions": top_fns,
                "mem_sections": mem_sections,
                "memory_analysis": memory_analysis,
            }
            all_runs.append(run_data)

    if os.path.exists(binary_path):
        try:
            os.remove(binary_path)
        except OSError:
            pass

    # --- Relatório ---
    txt_path = os.path.join(results_dir, "summary.txt")
    md_report = []
    md_report.append("==================================================")
    md_report.append(f" BitNet b1.58 -- Relatorio de Benchmark ({VERSION})")
    md_report.append("==================================================")
    md_report.append(f"Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_report.append("## Configuracao")
    md_report.append(f"  CPU:         {CPU_LABEL}")
    md_report.append(f"  Perf:        {'Disponivel' if perf_available else 'Indisponivel'}")
    md_report.append(f"  Estrategia:  Grupo unico ({len(PERF_EVENTS)} eventos, sem multiplexacao)")
    md_report.append(f"  Eventos:     {', '.join(PERF_EVENTS)}")
    md_report.append(f"  Cenarios:    {len(scenarios_to_run)}")
    md_report.append(f"  Compilacoes: {len(compilations_to_run)}")
    md_report.append("")

    for run in all_runs:
        md_report.append("=" * 50)
        md_report.append(f"### {run['compilation']} | {run['scenario']} ({run['scenario_id']})")
        md_report.append("=" * 50)
        if run.get("scenario_focus"):
            md_report.append(f"- Foco do cenario:     {run['scenario_focus']}")
        md_report.append(f"- Flags de compilacao: {run['compilation_flags']}")
        md_report.append(f"- Tokens de prompt:    {run['prompt_len']} (exato) | Tokens gerados: {run['gen_len']}")
        md_report.append(f"- IDs dos tokens:      {run['prompt_tokens']}")
        md_report.append(f"- Prompt:              \"{run['prompt_text']}\"")
        md_report.append(f"- Saída decodificada:  \"{run['output_text']}\"")
        md_report.append("")

        md_report.append("#### Timings internos (profiler C)")
        md_report.append("  Fase                                  | Tempo (s)    | % do Total")
        md_report.append("  " + "-" * 38 + "|" + "-" * 14 + "|" + "-" * 11)
        total_inf = run["c_timings"].get("Total Inference", 1.0)
        for key in sorted(run["c_timings"].keys()):
            val = run["c_timings"][key]
            pct = (val / total_inf * 100.0) if total_inf > 0 else 0.0
            md_report.append(f"  {key:<38} | {val:<12.6f} | {pct:<9.2f}%")
        md_report.append("")

        if perf_available:
            md_report.append("#### Perf -- Contadores (grupo unico, SEM multiplexacao)")
            md_report.append("  Evento                                | Valor")
            md_report.append("  " + "-" * 38 + "|" + "-" * 20)
            for k, v in run["perf_stats"].items():
                if isinstance(v, int):
                    md_report.append(f"  {k:<38} | {v:,}")
                elif isinstance(v, float):
                    md_report.append(f"  {k:<38} | {v:.2f}")
            md_report.append("")

            md_report.append("#### Perf -- Top 10 Funcoes por Tempo de CPU")
            md_report.append("  Overhead   | Simbolo")
            md_report.append("  " + "-" * 12 + "|" + "-" * 40)
            for fn in run["top_functions"]:
                md_report.append(f"  {fn['overhead']:<10} | {fn['symbol']}")
            md_report.append("")

            md_report.append("#### Perf Mem -- Latencia de loads/stores por nivel (PEBS)")
            if run.get("mem_sections"):
                for sec in run["mem_sections"]:
                    md_report.append(
                        f"  [{sec['kind'].upper()}] latencia media: "
                        f"{sec.get('avg_latency_ns', '?')} ns "
                        f"({sec.get('avg_latency_cycles', '?')} ciclos) | "
                        f"amostras totais: {sec['total_samples']:,}"
                    )
                    if sec["kind"] == "stores" and sec.get("avg_latency_cycles", 0) <= 1.5:
                        md_report.append(
                            "  AVISO: Latencia de stores ~1 ciclo -- "
                            "evento sem peso de latencia real neste hardware."
                        )
                    md_report.append("  Nivel          | % amostras | num amostras")
                    md_report.append("  " + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 14)
                    for lvl in sec["levels"]:
                        md_report.append(
                            f"  {lvl['level']:<14} | {lvl['overhead_pct']:<10.2f} | {lvl['samples']:,}"
                        )
                    md_report.append("")
            else:
                md_report.append("  (sem dados nesta execucao)")
            md_report.append("")

            md_report.append("#### Analise de Memoria -- Impacto na BitLinear")
            ma = run.get("memory_analysis", {})
            dist = ma.get("level_distribution", {})
            if dist:
                md_report.append("  Distribuicao de loads por nivel:")
                md_report.append("  Nivel  | % loads")
                md_report.append("  " + "-" * 8 + "|" + "-" * 10)
                for lv, pct in dist.items():
                    md_report.append(f"  {lv:<6} | {pct:>8.2f}%")
                md_report.append("")
            if ma.get("avg_latency_cycles") is not None:
                md_report.append(
                    f"  Latencia media loads : {ma['avg_latency_cycles']} ciclos "
                    f"/ {ma['avg_latency_ns']} ns"
                )
            if ma.get("mem_wait_fraction") is not None:
                md_report.append(
                    f"  Fracao de espera mem : {ma['mem_wait_fraction']:.4f} "
                    f"(L1 ideal = {L1_IDEAL_LATENCY_CYCLES} ciclos)"
                )
            if ma.get("bitlinear_overhead_pct") is not None:
                md_report.append(
                    f"  Overhead bitlinear   : {ma['bitlinear_overhead_pct']:.2f}% de CPU"
                )
                md_report.append(
                    f"  Tempo perdido c/ mem : {ma['time_lost_mem_pct']:.2f}% de CPU total"
                )
                md_report.append(
                    f"  Tempo efetivo comput : {ma['time_effective_compute_pct']:.2f}% de CPU total"
                )
            if not dist and ma.get("avg_latency_cycles") is None:
                md_report.append("  (sem dados de memoria nesta execucao)")
            md_report.append("")

    report_text = "\n".join(md_report)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    for fpath in [txt_path, results_dir]:
        restore_ownership(fpath)

    print(f"Relatorio texto salvo em:    {txt_path}")
    print(f"Log completo salvo em:       {log_path}")
    print("\n==================================================")
    print(" Benchmark concluido com sucesso!")
    print("==================================================")

    tee.close()
    restore_ownership(log_path)


if __name__ == "__main__":
    main()