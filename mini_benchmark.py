"""
mini_benchmark.py
=================
Mini-benchmark focado em acessos à hierarquia de memória (L1 → L2 → L3 → DRAM).

Hardware: AMD Ryzen 5 7520U (Zen 3+)
Objetivo: medir exatamente QUANTOS loads chegaram em cada nível de cache
          para BitNet V1 e V2, short prompt e long prompt, build de DEBUG.

Experimentos:
  1. V1 × Short prompt
  2. V1 × Long prompt
  3. V2 × Short prompt
  4. V2 × Long prompt

Compilação: -g -O0 -fno-inline -fno-inline-functions  (debug puro)

Eventos AMD Zen 3+ usados:
  ls_dc_accesses              — total de acessos ao L1-D (equivalente a L1 loads total)
  ls_refills_from_sys.ls_mabresp_lcl_l2   — refills do L1 vindos do L2 (L1 miss → L2 hit)
  ls_refills_from_sys.ls_mabresp_lcl_dram — refills do L1 vindos da DRAM LOCAL (L3 miss)
  ls_refills_from_sys.ls_mabresp_rmt_dram — refills vindos de DRAM REMOTA (NUMA)
  l2_cache_req_stat.ls_rd_blk_c — requisições de dados chegando ao L2 (L2 loads total)
  l2_pf_miss_l2_l3            — misses no L2 que chegaram ao L3 (L2 miss → L3 hit ou DRAM)
  l2_pf_miss_l2_hit_l3        — misses do L2 que HIT no L3

Derivado:
  DRAM_accesses ≈ ls_refills_from_sys.ls_mabresp_lcl_dram
                + ls_refills_from_sys.ls_mabresp_rmt_dram
  (= L3 miss para dados: cache lines novas vindas da DRAM física)

Nota do professor:
  "Use misses de L3 como aproximação de acessos à DRAM" →
  No AMD Zen, ls_refills_from_sys.ls_mabresp_lcl_dram é exatamente isso.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# ============================================================================
# Configuração
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()

# Ajuste para o diretório onde estão os .bin — mesmo diretório do projeto
DATA_DIR = BASE_DIR  # norms.bin, embeddings.bin etc. ficam aqui

CPU_LABEL = "AMD Ryzen 5 7520U (Zen 3+)"

# ============================================================================
# Prompts (mesmos do run_benchmark.py para comparação futura)
# ============================================================================

SHORT_PROMPT_TOKENS = [128000, 3923, 374, 279, 1925, 7580, 315, 30828, 14488, 30]
SHORT_PROMPT_TEXT   = "What is the main purpose of neural networks?"

LONG_PROMPT_TOKENS  = [
    128000, 6854, 499, 10552, 279, 1925, 12062, 1990, 5410, 19596,
    16983, 4211, 323, 72717, 661, 4211, 1093, 6631, 7099, 293, 16,
    13, 2970, 11, 11951, 21760, 389, 1268, 15449, 10484, 2065, 323,
    72717, 661, 14661, 8108, 5044, 34494, 8670, 323, 55580, 40370,
    2391, 12035, 19576, 389, 90562, 14511, 3046, 30,
]
LONG_PROMPT_TEXT = (
    "Can you explain the main differences between standard floating-point "
    "models and ternary models like BitNet b1.58?"
)

# Tokens gerados por experimento (igual ao run_benchmark.py)
MAX_GEN_SHORT = 10
MAX_GEN_LONG  = 50

# ============================================================================
# Build de debug
# ============================================================================

DEBUG_FLAGS_COMMON = ["-g", "-O0", "-fno-inline", "-fno-inline-functions", "-Wall"]

VERSIONS = [
    {"id": "v1", "version_num": 1, "weights_file": "weights.bin"},
    {"id": "v2", "version_num": 2, "weights_file": "weights_packed.bin"},
]

SCENARIOS = [
    {"id": "short", "label": "Short Prompt", "tokens": SHORT_PROMPT_TOKENS, "text": SHORT_PROMPT_TEXT, "max_gen": MAX_GEN_SHORT},
    {"id": "long",  "label": "Long Prompt",  "tokens": LONG_PROMPT_TOKENS,  "text": LONG_PROMPT_TEXT,  "max_gen": MAX_GEN_LONG},
]

# ============================================================================
# Eventos AMD Zen 3+
# Divididos em grupos de no máximo 4 para evitar multiplexação.
# AMD Zen 3 tipicamente tem 6 contadores PMU de propósito geral.
# ============================================================================

# Grupo 1: acessos totais ao L1-D e refills vindos do L2
PERF_GROUP_L1 = [
    "ls_dc_accesses",                          # Acessos totais ao L1-D (≈ loads L1)
    "ls_refills_from_sys.ls_mabresp_lcl_l2",   # Refills L1 ← L2 (L1 miss, L2 hit)
]

# Grupo 2: refills chegando da DRAM (= L3 miss ≈ acesso à DRAM)
PERF_GROUP_DRAM = [
    "ls_refills_from_sys.ls_mabresp_lcl_dram", # L1 refill ← DRAM local (L3 miss)
    "ls_refills_from_sys.ls_mabresp_rmt_dram", # L1 refill ← DRAM remota (NUMA)
    "ls_refills_from_sys.ls_mabresp_lcl_cache",# L1 refill ← outro cache (cross-core)
]

# Grupo 3: acessos e misses no L2
PERF_GROUP_L2 = [
    "l2_cache_req_stat.ls_rd_blk_c",           # Requisições de dados ao L2 (L2 loads)
    "l2_pf_miss_l2_hit_l3",                    # Miss L2, hit L3
    "l2_pf_miss_l2_l3",                        # Miss L2 (total — inclui DRAM)
]

# Grupo 4: overhead geral
PERF_GROUP_MAIN = [
    "cycles",
    "instructions",
    "cache-misses",                             # LLC misses (alias genérico)
    "cache-references",                         # LLC references
]

ALL_GROUPS = {
    "l1":   PERF_GROUP_L1,
    "dram": PERF_GROUP_DRAM,
    "l2":   PERF_GROUP_L2,
    "main": PERF_GROUP_MAIN,
}

# ============================================================================
# Helpers
# ============================================================================

def check_perf():
    try:
        r = subprocess.run(["perf", "--version"], capture_output=True)
        if r.returncode != 0:
            return False, "perf não instalado"
    except FileNotFoundError:
        return False, "perf não encontrado"

    paranoid = Path("/proc/sys/kernel/perf_event_paranoid")
    if paranoid.exists():
        level = int(paranoid.read_text().strip())
        if level > 1 and os.geteuid() != 0:
            return False, (
                f"perf_event_paranoid={level}. Execute com sudo ou:\n"
                "  sudo sysctl -w kernel.perf_event_paranoid=1"
            )
    return True, "OK"


def compile_binary(version_num: int, binary_path: str) -> bool:
    """Compila com BITNET_VERSION=version_num e flags de debug."""
    flags = [f"-DBITNET_VERSION={version_num}"] + DEBUG_FLAGS_COMMON
    cmd = (
        ["gcc"] + flags
        + [str(BASE_DIR / "bitnet.c"), str(BASE_DIR / "testbench.c")]
        + ["-o", binary_path, "-lm"]
    )
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [ERRO] Compilação falhou:\n{r.stderr}")
        return False
    return True


def run_perf_group(binary: str, tokens_str: str, weights_file: str,
                   events: list, max_gen: int) -> dict:
    """Executa o binário sob perf stat com um grupo de eventos."""
    env = os.environ.copy()
    env["MAX_GEN"] = str(max_gen)
    env["WEIGHTS_FILE"] = weights_file

    events_str = ",".join(events)
    cmd = (
        ["perf", "stat", "-e", events_str, "-x", ",", binary]
        + tokens_str.split()
    )

    r = subprocess.run(cmd, env=env, cwd=str(DATA_DIR),
                       capture_output=True, text=True)

    stats = {}
    for line in r.stderr.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        val_str = parts[0].strip().replace(",", "")
        event_name = parts[2].strip() if len(parts) > 2 else parts[1].strip()
        # Limpa sufixos como ":u" ou ":k" do nome
        event_name = event_name.split(":")[0].rstrip()
        try:
            stats[event_name] = int(val_str)
        except ValueError:
            # Pode ser "<not counted>" ou "<not supported>"
            stats[event_name] = val_str
    return stats


def run_experiment(binary: str, tokens: list, weights_file: str, max_gen: int) -> dict:
    """Roda todos os grupos de eventos e retorna resultados consolidados."""
    tokens_str = " ".join(str(t) for t in tokens)
    all_stats = {}
    for group_name, events in ALL_GROUPS.items():
        print(f"    Grupo [{group_name}]: {events}")
        group_stats = run_perf_group(binary, tokens_str, weights_file, events, max_gen)
        all_stats.update(group_stats)
    return all_stats


def derive_metrics(stats: dict) -> dict:
    """Calcula métricas derivadas."""
    l1_total   = stats.get("ls_dc_accesses", 0)
    l1_to_l2   = stats.get("ls_refills_from_sys.ls_mabresp_lcl_l2", 0)
    dram_local = stats.get("ls_refills_from_sys.ls_mabresp_lcl_dram", 0)
    dram_remote= stats.get("ls_refills_from_sys.ls_mabresp_rmt_dram", 0)
    cross_cache= stats.get("ls_refills_from_sys.ls_mabresp_lcl_cache", 0)
    l2_total   = stats.get("l2_cache_req_stat.ls_rd_blk_c", 0)
    l2_to_l3   = stats.get("l2_pf_miss_l2_hit_l3", 0)
    l2_miss_all= stats.get("l2_pf_miss_l2_l3", 0)
    cycles     = stats.get("cycles", 0)
    instrs     = stats.get("instructions", 0)
    llc_misses = stats.get("cache-misses", 0)
    llc_refs   = stats.get("cache-references", 0)

    derived = {}
    dram_total = 0
    if isinstance(dram_local, int): dram_total += dram_local
    if isinstance(dram_remote, int): dram_total += dram_remote
    derived["DRAM_accesses (L3 miss total)"] = dram_total

    if isinstance(l1_total, int) and l1_total > 0 and isinstance(l1_to_l2, int):
        derived["L1 miss rate (%)"] = round(l1_to_l2 / l1_total * 100, 4)

    if isinstance(l2_total, int) and l2_total > 0 and isinstance(l2_miss_all, int):
        derived["L2 miss rate (%)"] = round(l2_miss_all / l2_total * 100, 4)

    if isinstance(llc_refs, int) and llc_refs > 0 and isinstance(llc_misses, int):
        derived["LLC miss rate (%)"] = round(llc_misses / llc_refs * 100, 4)

    if isinstance(cycles, int) and cycles > 0 and isinstance(instrs, int):
        derived["IPC"] = round(instrs / cycles, 3)

    return derived


def fmt_val(v) -> str:
    if isinstance(v, int):
        return f"{v:>18,}"
    return f"{str(v):>18}"


def print_table(title: str, stats: dict, derived: dict):
    """Imprime tabela de resultados no terminal."""
    sep = "─" * 62
    print(f"\n  ┌{sep}┐")
    print(f"  │ {title:<60} │")
    print(f"  ├{sep}┤")
    print(f"  │ {'Evento':<42} {'Valor':>18} │")
    print(f"  ├{sep}┤")

    KEY_ORDER = [
        "ls_dc_accesses",
        "ls_refills_from_sys.ls_mabresp_lcl_l2",
        "l2_cache_req_stat.ls_rd_blk_c",
        "l2_pf_miss_l2_hit_l3",
        "l2_pf_miss_l2_l3",
        "ls_refills_from_sys.ls_mabresp_lcl_dram",
        "ls_refills_from_sys.ls_mabresp_rmt_dram",
        "ls_refills_from_sys.ls_mabresp_lcl_cache",
        "cache-references",
        "cache-misses",
        "cycles",
        "instructions",
    ]
    for k in KEY_ORDER:
        if k in stats:
            label = k[:42]
            print(f"  │ {label:<42} {fmt_val(stats[k])} │")

    print(f"  ├{sep}┤")
    print(f"  │ {'--- Derivados ---':<60} │")
    print(f"  ├{sep}┤")
    for k, v in derived.items():
        label = k[:42]
        val_str = f"{v:>18,}" if isinstance(v, int) else f"{str(v):>18}"
        print(f"  │ {label:<42} {val_str} │")
    print(f"  └{sep}┘")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 66)
    print(" Mini-Benchmark — Hierarquia de Memória BitNet (AMD Zen 3+)")
    print("=" * 66)
    print(f" CPU:       {CPU_LABEL}")
    print(f" Build:     DEBUG  ({' '.join(DEBUG_FLAGS_COMMON)})")
    print(f" Versões:   V1 (weights.bin) × V2 (weights_packed.bin)")
    print(f" Cenários:  Short prompt ({len(SHORT_PROMPT_TOKENS)} tokens, gera {MAX_GEN_SHORT} tokens)"
          f"  ×  Long prompt ({len(LONG_PROMPT_TOKENS)} tokens, gera {MAX_GEN_LONG} tokens)")
    print(f" Data dir:  {DATA_DIR}")
    print("=" * 66)

    # Verificar perf
    ok, msg = check_perf()
    if not ok:
        print(f"\n[ERRO] {msg}")
        sys.exit(1)
    print(f"\n[perf] {msg}")

    # Verificar arquivos de dados
    missing = []
    for f in ["norms.bin", "scales.bin", "embeddings.bin"]:
        if not (DATA_DIR / f).exists():
            missing.append(f)
    if missing:
        print(f"\n[AVISO] Arquivos ausentes em {DATA_DIR}: {missing}")
        print("  O testbench usará dados dummy (resultados não representarão inferência real).")

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = []
    exp_num = 0

    for ver in VERSIONS:
        binary = str(BASE_DIR / f"mini_bench_{ver['id']}")

        print(f"\n{'='*66}")
        print(f" Compilando {ver['id'].upper()} (BITNET_VERSION={ver['version_num']}, DEBUG)")
        print(f"{'='*66}")

        if not compile_binary(ver["version_num"], binary):
            print(f"  ⚠ Pulando {ver['id']} por erro de compilação.")
            continue

        # Verificar arquivo de pesos desta versão
        wfile = ver["weights_file"]
        if not (DATA_DIR / wfile).exists():
            print(f"  [AVISO] {wfile} não encontrado — usando dummy data.")

        for sc in SCENARIOS:
            exp_num += 1
            label = f"Exp {exp_num}/4 — {ver['id'].upper()} × {sc['label']}"
            print(f"\n{'─'*66}")
            print(f" {label}")
            print(f" Prompt: \"{sc['text'][:60]}\"")
            print(f" Tokens: {len(sc['tokens'])} tokens  |  MAX_GEN={sc['max_gen']}")
            print(f"{'─'*66}")

            t0 = time.monotonic()
            stats = run_experiment(binary, sc["tokens"], wfile, sc["max_gen"])
            elapsed = time.monotonic() - t0

            derived = derive_metrics(stats)
            print_table(label, stats, derived)
            print(f"\n  Tempo de coleta perf: {elapsed:.1f}s")

            all_results.append({
                "label":   label,
                "version": ver["id"],
                "scenario": sc["id"],
                "stats":   stats,
                "derived": derived,
            })

        # Limpa binário
        try:
            os.remove(binary)
        except OSError:
            pass

    # ── Relatório comparativo final ──────────────────────────────────────────
    print(f"\n\n{'='*66}")
    print(" COMPARATIVO FINAL — Acessos à Hierarquia de Memória")
    print(f"{'='*66}")

    COMPARE_KEYS = [
        ("L1 acessos totais",   "ls_dc_accesses"),
        ("L1→L2 refills",       "ls_refills_from_sys.ls_mabresp_lcl_l2"),
        ("L2 acessos totais",   "l2_cache_req_stat.ls_rd_blk_c"),
        ("L2→L3 misses",        "l2_pf_miss_l2_l3"),
        ("L3 miss → DRAM local","ls_refills_from_sys.ls_mabresp_lcl_dram"),
        ("L3 miss → DRAM remot","ls_refills_from_sys.ls_mabresp_rmt_dram"),
        ("DRAM total (aprox)",  "DRAM_accesses (L3 miss total)"),
        ("IPC",                 "IPC"),
    ]

    # Cabeçalho
    exp_labels = [r["label"].replace("Exp ", "").split("—")[1].strip() for r in all_results]
    col_w = 18
    header = f"  {'Métrica':<28}" + "".join(f"{lbl:>{col_w}}" for lbl in exp_labels)
    print(header)
    print("  " + "─" * (28 + col_w * len(all_results)))

    for display_name, key in COMPARE_KEYS:
        row = f"  {display_name:<28}"
        for r in all_results:
            # Busca nos stats brutos ou nos derivados
            v = r["stats"].get(key, r["derived"].get(key, "N/A"))
            if isinstance(v, int):
                row += f"{v:>{col_w},}"
            else:
                row += f"{str(v):>{col_w}}"
        print(row)

    # Salvar relatório em texto
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"mini_bench_{ts}.txt"
    lines = []
    lines.append(f"Mini-Benchmark — Hierarquia de Memória BitNet")
    lines.append(f"CPU: {CPU_LABEL}")
    lines.append(f"Build: DEBUG  {' '.join(DEBUG_FLAGS_COMMON)}")
    lines.append(f"Gerado em: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for r in all_results:
        lines.append(f"\n{'='*60}")
        lines.append(r["label"])
        lines.append("Estatísticas brutas (AMD PMU):")
        for k, v in r["stats"].items():
            lines.append(f"  {k:<50} {fmt_val(v)}")
        lines.append("Derivados:")
        for k, v in r["derived"].items():
            val_s = f"{v:,}" if isinstance(v, int) else str(v)
            lines.append(f"  {k:<50} {val_s:>18}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Relatório salvo em: {report_path}]")
    print("\n✅ Mini-benchmark concluído.")


if __name__ == "__main__":
    main()
