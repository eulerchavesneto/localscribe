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
import json
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path

import yt_dlp
from faster_whisper import WhisperModel

REFERER = "https://academy.desecsecurity.com/"


def slug(text, limit=60):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text[:limit] or "sem-titulo"


def hms(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


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
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return next(Path(dest_dir).glob("aula.*"))


def transcribe(model, path, language):
    segments, info = model.transcribe(
        str(path), language=language, vad_filter=True, beam_size=5
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
    ap.add_argument("--timestamps", action="store_true", help="inclui [hh:mm:ss] em cada trecho")
    ap.add_argument("--referer", default=REFERER, help="site do curso (o Panda só serve o vídeo com esse Referer)")
    ap.add_argument("--list", action="store_true", help="só lista os módulos e sai")
    args = ap.parse_args()

    modules = json.loads(Path(args.manifest).read_text())
    if args.list:
        for i, m in enumerate(modules, 1):
            n_video = sum(1 for l in m["lessons"] if l.get("video"))
            print(f"{i:2d}. {m['module']}  ({n_video}/{len(m['lessons'])} aulas com vídeo)")
        return

    wanted = set(args.modules) if args.modules else None
    out_root = Path(args.out)
    model = None

    for i, m in enumerate(modules, 1):
        if wanted and i not in wanted:
            continue
        mod_dir = out_root / f"{i:02d}_{slug(m['module'])}"
        mod_dir.mkdir(parents=True, exist_ok=True)
        done = []
        for j, lesson in enumerate(m["lessons"], 1):
            if not lesson.get("video"):
                continue
            target = mod_dir / f"{j:03d}_{slug(lesson['title'])}.md"
            label = f"[{i:02d}/{j:03d}] {lesson['title']}"
            if target.exists():
                print(f"= {label} (já feito)")
                done.append(target)
                continue
            if model is None:
                print(f"Carregando modelo {args.model}...")
                model = WhisperModel(args.model, device="auto", compute_type="int8")
            tmp = tempfile.mkdtemp(prefix="aula_")
            try:
                print(f"> {label}")
                media = download(lesson["lib"], lesson["video"], tmp, args.referer)
                segments, duration = transcribe(model, media, args.language)
                target.write_text(
                    lesson_markdown(m["module"], lesson["title"], segments, args.timestamps),
                    encoding="utf-8",
                )
                print(f"  ok ({duration / 60:.0f} min de áudio)")
                done.append(target)
            except Exception as exc:  # uma aula com erro não derruba o lote
                print(f"  ERRO: {exc}", file=sys.stderr)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if done:
            combined = mod_dir.with_suffix(".md")
            combined.write_text(
                "\n\n---\n\n".join(p.read_text(encoding="utf-8") for p in sorted(done)),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
