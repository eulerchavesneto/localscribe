#!/usr/bin/env python3
"""Transcreve em lote as aulas de um curso hospedado no Panda Video.

Lê um manifest.json (módulo -> aulas -> id do vídeo), baixa o stream HLS de cada
aula com yt-dlp, transcreve com faster-whisper e grava um .md por aula, mais um
arquivo único por módulo. É retomável: aulas já transcritas são puladas.

Uso:
    python course_transcribe.py manifest.json --list
    python course_transcribe.py manifest.json --modules 4 5 --model small
    python course_transcribe.py manifest.json            # tudo

O manifest.json é gerado por course_manifest.js (colado no console do navegador).
"""
import argparse
import importlib.util
import json
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path


def _bootstrap_cuda_dlls():
    """Põe as DLLs do cuBLAS/cuDNN (wheels nvidia-*) no PATH, como o localscribe.py faz.

    Precisa rodar antes de importar o ctranslate2; sem isso a GPU falha com
    "Library cublas64_12.dll is not found".
    """
    if sys.platform != "win32":
        return
    for pkg in ("cublas", "cudnn", "cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(f"nvidia.{pkg}")
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        bin_dir = Path(list(spec.submodule_search_locations)[0]) / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_bootstrap_cuda_dlls()

# --- Fallback de DNS (só para o host de vídeo) --------------------------------
#
# O DNS do roteador às vezes falha de forma intermitente para *.pandavideo.com.br
# ("getaddrinfo failed"). Se o resolvedor do sistema falhar, pergunta o IP a um DNS
# público por UDP e segue com o hostname original (o TLS continua validando o
# certificado). Não altera nenhuma configuração do sistema.
import random
import socket
import struct

FALLBACK_DNS = "1.1.1.1"
FALLBACK_DNS_SUFFIX = "pandavideo.com.br"
_dns_cache = {}


def _query_a_record(host, server=FALLBACK_DNS, timeout=5):
    query = struct.pack(">HHHHHH", random.randrange(65536), 0x0100, 1, 0, 0, 0)
    for label in host.rstrip(".").split("."):
        query += bytes([len(label)]) + label.encode("ascii")
    query += b"\0" + struct.pack(">HH", 1, 1)  # tipo A, classe IN
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(query, (server, 53))
        data = sock.recv(1500)

    def skip_name(pos):
        while True:
            size = data[pos]
            if size == 0:
                return pos + 1
            if size & 0xC0 == 0xC0:  # ponteiro de compressão: 2 bytes e acabou
                return pos + 2
            pos += 1 + size

    answers = struct.unpack(">H", data[6:8])[0]
    pos = skip_name(12) + 4  # pergunta: nome + tipo + classe
    ips = []
    for _ in range(answers):
        pos = skip_name(pos)
        rtype, _, _, rdlen = struct.unpack(">HHIH", data[pos:pos + 10])
        pos += 10
        if rtype == 1 and rdlen == 4:
            ips.append(socket.inet_ntoa(data[pos:pos + 4]))
        pos += rdlen
    return ips


_system_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_with_fallback(host, *args, **kwargs):
    try:
        return _system_getaddrinfo(host, *args, **kwargs)
    except socket.gaierror:
        if not (isinstance(host, str) and host.endswith(FALLBACK_DNS_SUFFIX)):
            raise
        ips = _dns_cache.get(host) or _query_a_record(host)
        if not ips:
            raise
        if host not in _dns_cache:
            print(f"  DNS do sistema falhou para {host}; usando {FALLBACK_DNS}", file=sys.stderr, flush=True)
        _dns_cache[host] = ips
        results = []
        for ip in ips:
            results += _system_getaddrinfo(ip, *args, **kwargs)
        return results


socket.getaddrinfo = _getaddrinfo_with_fallback

import yt_dlp
from faster_whisper import WhisperModel

REFERER = "https://academy.desecsecurity.com/"
DEFAULT_PROMPT = (
    "Aula de pentest e segurança ofensiva em português do Brasil. Termos: Nmap, Scapy, Wireshark, tcpdump, "
    "Metasploit, msfvenom, Meterpreter, Burp Suite, Kali Linux, Nessus, OpenVAS, Hydra, John the Ripper, Hashcat, "
    "Mimikatz, SQLMap, Nikto, Netcat, Python, Bash, PowerShell, Assembly, buffer overflow, shellcode, ASLR, DEP, "
    "EIP, ESP, SQL injection, XSS, CSRF, LFI, RFI, reverse shell, payload, exploit, SYN, ACK, SYN-ACK, RST, TCP, "
    "UDP, ICMP, HTTP, HTTPS, DNS, dport, sport, sr1, TTL, Active Directory, SMB, Kerberos, NTLM, OSINT, VLAB."
)
# Os hotwords puxam a decodificação para estes termos com mais força que o prompt.
DEFAULT_HOTWORDS = "Scapy, Wireshark, Nmap, Metasploit, SYN, ACK, SYN-ACK, dport, sport, sr1"


def slug(text, limit=60):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:limit] or "sem-titulo"


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


PAUSE_BETWEEN_LESSONS = 5  # segundos; não martela o servidor do curso
NETWORK_RETRY_WAITS = [30, 60, 120, 300, 600]
NETWORK_ERROR_HINTS = ("getaddrinfo", "name resolution", "timed out", "timeout", "connection", "temporarily")


def download(lib, video, dest_dir, referer):
    url = f"https://b-{lib}.tv.pandavideo.com.br/{video}/playlist.m3u8"
    opts = {
        "format": "worst",  # a trilha de áudio é a mesma; 360p é o menor download
        "outtmpl": str(Path(dest_dir) / "aula.%(ext)s"),
        "http_headers": {"Referer": referer},
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
    }
    for attempt, wait in enumerate(NETWORK_RETRY_WAITS + [None], 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            break
        except Exception as exc:
            # Só repete falha de rede (DNS, timeout, conexão). Erro de acesso (401/403/429) não.
            if wait is None or not any(t in str(exc).lower() for t in NETWORK_ERROR_HINTS):
                raise
            print(f"  rede instável ({attempt}ª tentativa), aguardando {wait}s...", file=sys.stderr, flush=True)
            time.sleep(wait)
    return next(Path(dest_dir).glob("aula.*"))


def prefetch(items, referer, ready):
    """Baixa uma aula por vez, na ordem, e entrega na fila; None marca o fim."""
    for item in items:
        tmp = tempfile.mkdtemp(prefix="aula_")
        try:
            ready.put((item, tmp, download(item["lib"], item["video"], tmp, referer), None))
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            ready.put((item, None, None, exc))
        time.sleep(PAUSE_BETWEEN_LESSONS)
    ready.put(None)


def write_combined(mod_dir):
    """Junta as aulas já transcritas do módulo em um único .md."""
    lessons = sorted(mod_dir.glob("[0-9][0-9][0-9]_*.md"))
    if lessons:
        mod_dir.with_suffix(".md").write_text(
            "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in lessons), encoding="utf-8"
        )


def transcribe(model, path, language, prompt, hotwords=None):
    segments, info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        beam_size=5,
        initial_prompt=prompt or None,
        hotwords=hotwords or None,
        condition_on_previous_text=False,  # evita laços de repetição em trechos longos
    )
    return [(s.start, s.text.strip()) for s in segments], info.duration


def lesson_markdown(module, title, segments, timestamps):
    lines = [f"# {title}", f"*Módulo: {module}*", ""]
    if timestamps:
        lines += [f"[{hms(start)}] {text}" for start, text in segments]
    else:
        lines.append(" ".join(text for _, text in segments))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", default="transcricoes")
    ap.add_argument("--modules", nargs="*", type=int, help="números dos módulos (veja --list); padrão: todos")
    ap.add_argument("--model", default="small", help="tiny, base, small, medium, large-v3-turbo...")
    ap.add_argument("--language", default="pt")
    ap.add_argument("--compute-type", default="auto", help="auto, float16 (GPU, mais preciso), int8 (CPU)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="vocabulário que ajuda o Whisper com termos técnicos")
    ap.add_argument("--hotwords", default=DEFAULT_HOTWORDS, help="termos que o Whisper deve preferir (vazio desliga)")
    ap.add_argument("--timestamps", action="store_true", help="inclui [hh:mm:ss] em cada trecho")
    ap.add_argument("--referer", default=REFERER, help="site do curso (o Panda só serve o vídeo com esse Referer)")
    ap.add_argument("--list", action="store_true", help="só lista os módulos e sai")
    args = ap.parse_args()

    modules = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.list:
        for i, m in enumerate(modules, 1):
            n_video = sum(1 for l in m["lessons"] if l.get("video"))
            print(f"{i:2d}. {m['module']}  ({n_video}/{len(m['lessons'])} aulas com vídeo)")
        return

    wanted = set(args.modules) if args.modules else None
    out_root = Path(args.out)

    pending, skipped = [], 0
    for i, m in enumerate(modules, 1):
        if wanted and i not in wanted:
            continue
        mod_dir = out_root / f"{i:02d}_{slug(m['module'])}"
        mod_dir.mkdir(parents=True, exist_ok=True)
        for j, lesson in enumerate(m["lessons"], 1):
            if not lesson.get("video"):
                continue
            target = mod_dir / f"{j:03d}_{slug(lesson['title'])}.md"
            if target.exists():
                skipped += 1
                continue
            pending.append({
                "label": f"[{i:02d}/{j:03d}] {lesson['title']}", "module": m["module"],
                "title": lesson["title"], "lib": lesson["lib"], "video": lesson["video"],
                "target": target, "mod_dir": mod_dir,
            })
    print(f"{skipped} aulas já feitas, {len(pending)} pendentes")
    if not pending:
        return

    print(f"Carregando modelo {args.model}...")
    model = WhisperModel(args.model, device="auto", compute_type=args.compute_type)

    # Baixa a próxima aula enquanto a GPU transcreve a atual. Continua sendo um download
    # por vez, com pausa entre eles; a fila de tamanho 1 impede de adiantar mais que isso.
    ready = queue.Queue(maxsize=1)
    threading.Thread(target=prefetch, args=(pending, args.referer, ready), daemon=True).start()

    while (got := ready.get()) is not None:
        item, tmp, media, error = got
        print(f"> {item['label']}", flush=True)
        try:
            if error:
                raise error
            segments, duration = transcribe(model, media, args.language, args.prompt, args.hotwords)
            item["target"].write_text(
                lesson_markdown(item["module"], item["title"], segments, args.timestamps),
                encoding="utf-8",
            )
            print(f"  ok ({duration / 60:.0f} min de áudio)", flush=True)
            write_combined(item["mod_dir"])
        except Exception as exc:  # uma aula com erro não derruba o lote
            print(f"  ERRO: {exc}", file=sys.stderr, flush=True)
        finally:
            if tmp:
                shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
