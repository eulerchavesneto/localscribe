# coding: utf-8
"""
Aplicativo de Transcrição de Áudio e Vídeo (v2 — aceleração real por GPU)

Ferramenta focada em transcrever ficheiros de áudio/vídeo locais ou da internet
usando Whisper, executando 100% localmente.

Novidades da v2:
- Backend `faster-whisper` (CTranslate2): ~4x mais rápido e muito menos VRAM que
  o `openai-whisper`, sem depender do PyTorch.
- Aceleração por GPU que realmente funciona: detecção de CUDA, escolha automática
  de precisão (float16 / int8_float16) conforme a VRAM disponível, e diagnóstico
  claro quando a GPU não pode ser usada.
- Filtro VAD (detecção de voz) para pular silêncio e reduzir alucinações.
- Progresso real durante a transcrição (não mais barra indeterminada).
- Exportação em .txt, .srt, .vtt e .json.
- Cancelamento de lote em andamento.
- yt-dlp resiliente: playlists, mensagens de erro compreensíveis para bloqueio
  anti-bot do YouTube, e suporte a cookies do navegador.
- FFmpeg deixou de ser obrigatório (a decodificação é feita via PyAV).
"""

import ctypes
import gc
import importlib.util
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Bootstrap das DLLs CUDA (precisa rodar ANTES de importar ctranslate2) ---
#
# Os wheels `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` instalam as DLLs dentro do
# site-packages, mas não as colocam no PATH. O CTranslate2 as carrega via
# LoadLibrary, que consulta o PATH — `os.add_dll_directory` sozinho não resolve.

CUDA_DLL_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc")
REQUIRED_CUDA_DLLS = ("cublas64_12.dll", "cudnn64_9.dll")


def _bootstrap_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    for pkg in CUDA_DLL_PACKAGES:
        bin_dir = _nvidia_package_bin(pkg)
        if bin_dir is None:
            continue
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _nvidia_package_bin(pkg: str) -> Optional[Path]:
    """Localiza o diretório `bin` de um wheel nvidia-*, em venv ou instalação global."""
    try:
        spec = importlib.util.find_spec(f"nvidia.{pkg}")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        candidate = Path(list(spec.submodule_search_locations)[0]) / "bin"
        if candidate.is_dir():
            return candidate
    fallback = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / pkg / "bin"
    return fallback if fallback.is_dir() else None


_bootstrap_cuda_dlls()


# --- Verificação de dependências (sem instalação automática) ---
#
# A v1 instalava as dependências sozinha com `pip install torch`, o que no Windows
# baixa a wheel CPU-only e fazia a "aceleração por GPU" nunca funcionar. Agora a
# instalação é explícita e documentada no requirements.txt.

REQUIRED_MODULES = {
    "faster_whisper": "faster-whisper",
    "yt_dlp": "yt-dlp",
    "tkinterdnd2": "tkinterdnd2",
}


def check_dependencies() -> None:
    missing = [pkg for mod, pkg in REQUIRED_MODULES.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(
        "Dependências em Falta",
        "Os seguintes pacotes não estão instalados:\n\n"
        f"    {', '.join(missing)}\n\n"
        "Instale com:\n\n"
        f"    {Path(sys.executable).name} -m pip install -r requirements.txt\n\n"
        f"Interpretador em uso:\n{sys.executable}",
    )
    sys.exit(1)


check_dependencies()

import tkinterdnd2  # noqa: E402
import yt_dlp  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402


# --- Detecção de dispositivo -------------------------------------------------

@dataclass
class DeviceInfo:
    device: str          # "cuda" ou "cpu"
    gpu_name: str = ""
    total_vram_mb: int = 0
    problem: str = ""    # motivo pelo qual a GPU não pôde ser usada

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def usable_vram_mb(self) -> int:
        """VRAM total menos uma folga para o desktop/navegador."""
        return max(self.total_vram_mb - 600, 0)

    def describe(self) -> str:
        if self.is_cuda:
            return f"GPU: {self.gpu_name} — {self.total_vram_mb / 1024:.1f} GB VRAM"
        return f"CPU ({os.cpu_count() or '?'} núcleos)" + (f" — {self.problem}" if self.problem else "")


def _run_hidden(args: List[str]) -> str:
    """Executa um comando sem abrir janela de console no Windows."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15,
                             creationflags=flags)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _query_gpu() -> Tuple[str, int]:
    out = _run_hidden(["nvidia-smi", "--query-gpu=name,memory.total",
                       "--format=csv,noheader,nounits"])
    if not out:
        return "GPU CUDA", 0
    name, _, mem = out.splitlines()[0].partition(",")
    try:
        return name.strip(), int(mem.strip())
    except ValueError:
        return name.strip(), 0


def detect_device() -> DeviceInfo:
    """Descobre se dá para usar CUDA e, se não der, explica por quê."""
    import ctranslate2

    try:
        cuda_count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # pragma: no cover - depende do ambiente
        return DeviceInfo("cpu", problem=f"CTranslate2 falhou ao consultar CUDA: {exc}")

    if cuda_count < 1:
        return DeviceInfo("cpu", problem="nenhuma GPU NVIDIA detectada")

    # O CTranslate2 só carrega cuBLAS/cuDNN na primeira inferência; sem esta
    # checagem o usuário só descobriria o problema no meio de uma transcrição.
    if sys.platform == "win32":
        for dll in REQUIRED_CUDA_DLLS:
            try:
                ctypes.WinDLL(dll)
            except OSError:
                return DeviceInfo(
                    "cpu",
                    problem=(f"GPU detectada, mas {dll} não foi encontrada. "
                             "Instale: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"),
                )

    name, vram = _query_gpu()
    return DeviceInfo("cuda", gpu_name=name, total_vram_mb=vram)


# --- Catálogo de modelos -----------------------------------------------------
#
# VRAM em MiB, medida numa RTX 3050 4GB (pesos + ativações, beam_size=5).
# Valores marcados com ~ são extrapolados dos medidos.

MODELS: Dict[str, Dict[str, int]] = {
    "tiny":            {"float16": 400,  "int8_float16": 300},   # ~
    "base":            {"float16": 500,  "int8_float16": 350},   # ~
    "small":           {"float16": 727,  "int8_float16": 455},   # medido
    "medium":          {"float16": 2079, "int8_float16": 1087},  # medido
    "distil-large-v3": {"float16": 1600, "int8_float16": 950},   # ~
    "large-v3-turbo":  {"float16": 2215, "int8_float16": 1199},  # medido
    "large-v2":        {"float16": 3735, "int8_float16": 2071},  # ~ (igual v3)
    "large-v3":        {"float16": 3735, "int8_float16": 2071},  # medido
}

DEFAULT_MODEL = "large-v3-turbo"

CPU_COMPUTE_TYPES = ["int8", "float32"]
CUDA_COMPUTE_TYPES = ["float16", "int8_float16", "int8", "float32"]


def auto_compute_type(model: str, dev: DeviceInfo) -> str:
    """Escolhe a precisão mais rápida que ainda cabe na VRAM disponível."""
    if not dev.is_cuda:
        return "int8"
    budget = dev.usable_vram_mb
    needs = MODELS.get(model, MODELS[DEFAULT_MODEL])
    if budget == 0 or needs["float16"] <= budget:
        return "float16"
    if needs["int8_float16"] <= budget:
        return "int8_float16"
    return "int8_float16"  # melhor esforço; a UI já terá avisado


def vram_estimate_mb(model: str, compute_type: str) -> int:
    needs = MODELS.get(model)
    if not needs:
        return 0
    return needs.get(compute_type, needs["float16"])


# --- Formatos de saída -------------------------------------------------------

@dataclass
class Segment:
    start: float
    end: float
    text: str


def _timestamp(seconds: float, sep: str) -> str:
    ms = max(int(round(seconds * 1000)), 0)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def to_txt(segments: List[Segment], text: str) -> str:
    return text + "\n"


def to_srt(segments: List[Segment], text: str) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(f"{i}\n"
                      f"{_timestamp(seg.start, ',')} --> {_timestamp(seg.end, ',')}\n"
                      f"{seg.text}\n")
    return "\n".join(blocks)


def to_vtt(segments: List[Segment], text: str) -> str:
    blocks = ["WEBVTT\n"]
    for seg in segments:
        blocks.append(f"{_timestamp(seg.start, '.')} --> {_timestamp(seg.end, '.')}\n"
                      f"{seg.text}\n")
    return "\n".join(blocks)


def to_json(segments: List[Segment], text: str) -> str:
    payload = {
        "text": text,
        "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
                     for s in segments],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


WRITERS: Dict[str, Callable[[List[Segment], str], str]] = {
    "txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json,
}


# --- Backends de transcrição -------------------------------------------------

class Cancelled(Exception):
    """Sinaliza que o usuário abortou o processamento."""


@dataclass
class TranscriptionResult:
    text: str
    segments: List[Segment]
    language: str
    duration: float


class FasterWhisperBackend:
    """Backend padrão: CTranslate2, sem PyTorch."""

    name = "faster-whisper"

    def __init__(self) -> None:
        self._model: Optional[WhisperModel] = None
        self._key: Tuple[str, str, str] = ("", "", "")

    def load(self, model_name: str, device: str, compute_type: str) -> None:
        key = (model_name, device, compute_type)
        if self._model is not None and key == self._key:
            return
        self.unload()  # nunca manter dois modelos na VRAM ao mesmo tempo
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._key = key

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._key = ("", "", "")
            gc.collect()

    def transcribe(self, path: Path, language: Optional[str], task: str, vad: bool,
                   on_progress: Callable[[float], None],
                   cancel: threading.Event) -> TranscriptionResult:
        assert self._model is not None, "load() deve ser chamado antes de transcribe()"
        segment_iter, info = self._model.transcribe(
            str(path),
            language=language,
            task=task,
            beam_size=5,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # reduz repetições em áudio longo
        )
        collected: List[Segment] = []
        total = info.duration or 0.0
        for seg in segment_iter:
            if cancel.is_set():
                raise Cancelled()
            collected.append(Segment(seg.start, seg.end, seg.text.strip()))
            if total > 0:
                on_progress(min(seg.end / total * 100.0, 100.0))
        on_progress(100.0)
        text = " ".join(s.text for s in collected).strip()
        return TranscriptionResult(text, collected, info.language or "", total)


class OpenAIWhisperBackend:
    """Backend legado, opcional. Só aparece na UI se `openai-whisper` estiver instalado."""

    name = "openai-whisper"

    def __init__(self) -> None:
        self._model: Any = None
        self._key: Tuple[str, str] = ("", "")

    @staticmethod
    def available() -> bool:
        return (importlib.util.find_spec("whisper") is not None
                and importlib.util.find_spec("torch") is not None)

    def load(self, model_name: str, device: str, compute_type: str) -> None:
        import whisper
        key = (model_name, device)
        if self._model is not None and key == self._key:
            return
        self.unload()
        self._model = whisper.load_model(model_name, device=device)
        self._key = key

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._key = ("", "")
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def transcribe(self, path: Path, language: Optional[str], task: str, vad: bool,
                   on_progress: Callable[[float], None],
                   cancel: threading.Event) -> TranscriptionResult:
        assert self._model is not None, "load() deve ser chamado antes de transcribe()"
        result = self._model.transcribe(
            str(path), language=language, task=task,
            fp16=(self._key[1] == "cuda"),  # fp16 só faz sentido em GPU
        )
        if cancel.is_set():
            raise Cancelled()
        segments = [Segment(s["start"], s["end"], s["text"].strip())
                    for s in result.get("segments", [])]
        on_progress(100.0)
        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(result["text"].strip(), segments,
                                   result.get("language", ""), duration)


# --- Download de mídia (yt-dlp) ----------------------------------------------

URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

YDL_ERROR_HINTS = (
    (("sign in to confirm", "not a bot", "confirm you"),
     "O YouTube exigiu verificação anti-bot. Selecione um navegador em "
     "'Cookies do navegador' nas configurações e faça login no YouTube nele."),
    (("http error 403", "forbidden"),
     "Acesso negado (403). Isso normalmente é o anti-bot do YouTube: use a opção "
     "'Cookies do navegador', ou atualize o yt-dlp (pip install -U yt-dlp)."),
    (("private video", "members-only", "join this channel"),
     "Vídeo privado ou exclusivo para membros. É preciso usar cookies de uma conta com acesso."),
    (("video unavailable", "removed by the uploader"),
     "Vídeo indisponível ou removido."),
    (("age-restricted", "age restricted", "inappropriate"),
     "Vídeo com restrição de idade. Use 'Cookies do navegador' com uma conta logada."),
    (("unable to download webpage", "getaddrinfo", "timed out", "connection"),
     "Falha de rede ao contactar o servidor. Verifique sua conexão."),
)


def friendly_download_error(exc: Exception) -> str:
    raw = str(exc)
    low = raw.lower()
    for needles, hint in YDL_ERROR_HINTS:
        if any(n in low for n in needles):
            return hint
    return raw.strip() or "Falha desconhecida no download."


def build_ydl_opts(temp_dir: Path, cookies_browser: str,
                   hook: Optional[Callable] = None) -> Dict[str, Any]:
    """Baixa só a faixa de áudio, sem pós-processamento.

    Não há postprocessor FFmpegExtractAudio: o `faster-whisper` decodifica m4a/webm
    direto via PyAV e já reamostra para 16 kHz mono em memória. Isso dispensa o
    FFmpeg e evita escrever um WAV gigante em disco.
    """
    opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(temp_dir / "%(id)s.%(ext)s"),  # id é curto e sempre válido
        "noplaylist": True,  # playlists são expandidas explicitamente na UI
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,  # o progresso vai para a barra da UI, não para o stdout
        "retries": 5,
        "fragment_retries": 5,
        "socket_timeout": 30,
        "nocheckcertificate": False,
    }
    if hook is not None:
        opts["progress_hooks"] = [hook]
    if cookies_browser and cookies_browser != "Nenhum":
        opts["cookiesfrombrowser"] = (cookies_browser.lower(),)
    return opts


def downloaded_filepath(ydl: "yt_dlp.YoutubeDL", info: Dict[str, Any]) -> Path:
    """Caminho real do ficheiro baixado.

    `prepare_filename()` adivinha o nome; `requested_downloads` é o que o yt-dlp
    de facto escreveu — a diferença aparece sempre que há mudança de extensão.
    """
    downloads = info.get("requested_downloads")
    if downloads and downloads[0].get("filepath"):
        return Path(downloads[0]["filepath"])
    return Path(ydl.prepare_filename(info))


def safe_output_stem(title: str, fallback: str) -> str:
    stem = yt_dlp.utils.sanitize_filename(title or fallback, restricted=False)
    stem = stem.strip().rstrip(".")  # Windows rejeita nomes terminados em ponto
    return (stem[:120] or fallback)


def expand_playlist(url: str) -> List[str]:
    """Resolve uma URL de playlist na lista de URLs dos vídeos (sem baixar nada)."""
    opts = {"extract_flat": "in_playlist", "quiet": True, "no_warnings": True,
            "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or []
    urls = []
    for entry in entries:
        if not entry:
            continue
        target = entry.get("url") or entry.get("webpage_url") or entry.get("id")
        if target and not URL_PATTERN.match(target):
            target = f"https://www.youtube.com/watch?v={target}"
        if target:
            urls.append(target)
    return urls


# --- Persistência de preferências --------------------------------------------

SETTINGS_FILE = Path.home() / ".transcription_semai.json"


def load_settings() -> Dict[str, Any]:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(data: Dict[str, Any]) -> None:
    try:
        SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass  # preferências são um luxo; nunca devem quebrar o app


# --- Estado das tarefas ------------------------------------------------------

@dataclass
class Task:
    source: Any            # Path (ficheiro local) ou str (URL)
    display: str
    status: str = "Na Fila"
    text: str = ""
    error: str = ""
    saved: List[Path] = field(default_factory=list)

    @property
    def is_url(self) -> bool:
        return isinstance(self.source, str) and bool(URL_PATTERN.match(self.source))


# --- Aplicação ---------------------------------------------------------------

class AudioTranscriberApp:
    LANGUAGES = {
        "Detectar Automaticamente": None, "Português": "pt", "Inglês": "en",
        "Espanhol": "es", "Francês": "fr", "Alemão": "de", "Italiano": "it",
        "Japonês": "ja", "Chinês": "zh", "Russo": "ru", "Coreano": "ko",
        "Árabe": "ar", "Holandês": "nl", "Polaco": "pl", "Turco": "tr",
    }
    COOKIE_BROWSERS = ["Nenhum", "Chrome", "Firefox", "Edge", "Brave", "Opera", "Vivaldi"]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transcrição de Áudio e Vídeo — Whisper (v2)")
        self.root.minsize(900, 760)
        self.root.geometry("1100x820")

        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.style.configure("Accent.TButton", font=("Arial", 12, "bold"))
        self.style.configure("Treeview.Heading", font=("Arial", 10, "bold"))
        self.style.configure("Device.TLabel", font=("Arial", 10, "bold"))
        self.style.configure("Warn.TLabel", foreground="#b34700")
        self.style.configure("Ok.TLabel", foreground="#2e7d32")

        self.settings = load_settings()
        self.device_info = detect_device()
        self.backend: Any = FasterWhisperBackend()

        self.output_directory = Path(self.settings.get("output_dir", ""))
        self.task_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
        self.is_processing = False
        self.cancel_event = threading.Event()
        self.ffmpeg_path = Path()

        self._setup_ffmpeg_environment()

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.results_frame = ttk.Frame(self.root, padding="10")
        self.controls_to_toggle: List[tk.Widget] = []

        self._create_main_widgets()
        self._create_results_widgets()
        self._create_status_bar()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_queue)
        self.show_main_screen()
        self._refresh_estimate()

    # -- construção da UI ----------------------------------------------------

    def _create_main_widgets(self) -> None:
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        device_bar = ttk.Frame(self.main_frame, padding=(10, 4))
        device_bar.pack(fill=tk.X, padx=10)
        icon = "⚡" if self.device_info.is_cuda else "⏱"
        style = "Ok.TLabel" if self.device_info.is_cuda else "Warn.TLabel"
        ttk.Label(device_bar, text=f"{icon} Dispositivo: {self.device_info.describe()}",
                  style=style).pack(side=tk.LEFT)
        if not self.device_info.is_cuda:
            ttk.Button(device_bar, text="Por que não estou usando a GPU?",
                       command=self._show_gpu_help).pack(side=tk.LEFT, padx=10)

        task_frame = ttk.LabelFrame(self.main_frame, text="1. Lista de Tarefas", padding="10")
        task_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        buttons = ttk.Frame(task_frame)
        buttons.pack(fill=tk.X, pady=(0, 5))
        btn_files = ttk.Button(buttons, text="Adicionar Ficheiros...", command=self._browse_files)
        btn_files.pack(side=tk.LEFT)
        btn_urls = ttk.Button(buttons, text="Adicionar URLs...", command=self._add_urls_dialog)
        btn_urls.pack(side=tk.LEFT, padx=10)
        btn_remove = ttk.Button(buttons, text="Remover Selecionadas", command=self._remove_selected)
        btn_remove.pack(side=tk.LEFT)
        btn_clear = ttk.Button(buttons, text="Limpar Lista", command=self._clear_task_list)
        btn_clear.pack(side=tk.RIGHT)
        self.controls_to_toggle += [btn_files, btn_urls, btn_remove, btn_clear]

        self.task_tree = ttk.Treeview(task_frame, columns=("source", "status"),
                                      show="headings", height=8)
        self.task_tree.heading("source", text="Ficheiro ou URL")
        self.task_tree.heading("status", text="Status")
        self.task_tree.column("source", width=700)
        self.task_tree.column("status", width=170, anchor="center")
        self.task_tree.pack(fill=tk.BOTH, expand=True)
        self.task_tree.drop_target_register("DND_Files")
        self.task_tree.dnd_bind("<<Drop>>", self._handle_drop)

        config = ttk.LabelFrame(self.main_frame, text="2. Configurações", padding="10")
        config.pack(fill=tk.X, padx=10, pady=5)
        grid = ttk.Frame(config)
        grid.pack(fill=tk.X)

        ttk.Label(grid, text="Modelo:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        self.model_var = tk.StringVar(value=self.settings.get("model", DEFAULT_MODEL))
        model_combo = ttk.Combobox(grid, textvariable=self.model_var,
                                   values=list(MODELS), state="readonly", width=18)
        model_combo.grid(row=0, column=1, padx=5, pady=4, sticky="ew")
        model_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_estimate())

        ttk.Label(grid, text="Precisão:").grid(row=0, column=2, padx=(15, 5), pady=4, sticky="w")
        self.compute_var = tk.StringVar(value=self.settings.get("compute_type", "Automático"))
        allowed = CUDA_COMPUTE_TYPES if self.device_info.is_cuda else CPU_COMPUTE_TYPES
        self.compute_combo = ttk.Combobox(grid, textvariable=self.compute_var,
                                          values=["Automático"] + allowed,
                                          state="readonly", width=15)
        self.compute_combo.grid(row=0, column=3, padx=5, pady=4, sticky="ew")
        self.compute_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_estimate())
        if self.compute_var.get() not in ["Automático"] + allowed:
            self.compute_var.set("Automático")

        self.estimate_label = ttk.Label(grid, text="")
        self.estimate_label.grid(row=0, column=4, padx=15, pady=4, sticky="w")

        ttk.Label(grid, text="Idioma:").grid(row=1, column=0, padx=5, pady=4, sticky="w")
        self.lang_var = tk.StringVar(value=self.settings.get("language", "Detectar Automaticamente"))
        lang_combo = ttk.Combobox(grid, textvariable=self.lang_var,
                                  values=list(self.LANGUAGES), state="readonly", width=18)
        lang_combo.grid(row=1, column=1, padx=5, pady=4, sticky="ew")

        self.translate_var = tk.BooleanVar(value=False)
        translate_check = ttk.Checkbutton(grid, text="Traduzir para Inglês",
                                          variable=self.translate_var)
        translate_check.grid(row=1, column=2, columnspan=2, padx=(15, 5), pady=4, sticky="w")

        self.vad_var = tk.BooleanVar(value=self.settings.get("vad", True))
        vad_check = ttk.Checkbutton(grid, text="Filtro de voz (VAD) — pula silêncio",
                                    variable=self.vad_var)
        vad_check.grid(row=1, column=4, padx=15, pady=4, sticky="w")

        ttk.Label(grid, text="Backend:").grid(row=2, column=0, padx=5, pady=4, sticky="w")
        backends = ["faster-whisper"]
        if OpenAIWhisperBackend.available():
            backends.append("openai-whisper")
        self.backend_var = tk.StringVar(value="faster-whisper")
        backend_combo = ttk.Combobox(grid, textvariable=self.backend_var, values=backends,
                                     state="readonly", width=18)
        backend_combo.grid(row=2, column=1, padx=5, pady=4, sticky="ew")
        backend_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_estimate())

        ttk.Label(grid, text="Cookies do navegador:").grid(row=2, column=2, padx=(15, 5),
                                                          pady=4, sticky="w")
        self.cookies_var = tk.StringVar(value=self.settings.get("cookies", "Nenhum"))
        cookies_combo = ttk.Combobox(grid, textvariable=self.cookies_var,
                                     values=self.COOKIE_BROWSERS, state="readonly", width=15)
        cookies_combo.grid(row=2, column=3, padx=5, pady=4, sticky="ew")

        formats_frame = ttk.Frame(grid)
        formats_frame.grid(row=2, column=4, padx=15, pady=4, sticky="w")
        ttk.Label(formats_frame, text="Salvar como:").pack(side=tk.LEFT)
        self.format_vars: Dict[str, tk.BooleanVar] = {}
        saved_formats = self.settings.get("formats", ["txt"])
        for fmt in WRITERS:
            var = tk.BooleanVar(value=fmt in saved_formats)
            self.format_vars[fmt] = var
            chk = ttk.Checkbutton(formats_frame, text=f".{fmt}", variable=var)
            chk.pack(side=tk.LEFT, padx=2)
            self.controls_to_toggle.append(chk)

        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)
        self.controls_to_toggle += [model_combo, self.compute_combo, lang_combo,
                                    translate_check, vad_check, backend_combo, cookies_combo]

        output = ttk.LabelFrame(self.main_frame, text="3. Pasta de Destino", padding="10")
        output.pack(fill=tk.X, padx=10, pady=5)
        self.output_dir_var = tk.StringVar(
            value=str(self.output_directory) if self.output_directory.is_dir()
            else "Nenhuma pasta selecionada."
        )
        ttk.Entry(output, textvariable=self.output_dir_var, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        btn_folder = ttk.Button(output, text="Selecionar Pasta...",
                                command=self._select_output_directory)
        btn_folder.pack(side=tk.LEFT)
        self.controls_to_toggle.append(btn_folder)

        actions = ttk.Frame(self.main_frame)
        actions.pack(fill=tk.X, padx=10, pady=15)
        self.start_button = ttk.Button(actions, text="INICIAR PROCESSAMENTO",
                                       command=self._start_processing, style="Accent.TButton")
        self.start_button.pack(side=tk.RIGHT, ipady=10, ipadx=10)
        self.cancel_button = ttk.Button(actions, text="Cancelar", command=self._cancel_processing,
                                        state="disabled")
        self.cancel_button.pack(side=tk.RIGHT, padx=10, ipady=10)
        self.controls_to_toggle.append(self.start_button)

    def _create_results_widgets(self) -> None:
        toolbar = ttk.Frame(self.results_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(toolbar, text="Salvar Como...", command=self._save_current).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Copiar", command=self._copy_to_clipboard).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="Abrir Pasta de Destino",
                   command=self._open_output_dir).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="← Voltar", command=self.show_main_screen).pack(side=tk.RIGHT)

        panes = ttk.PanedWindow(self.results_frame, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(panes)
        self.results_list = tk.Listbox(left, exportselection=False, font=("Arial", 10))
        self.results_list.pack(fill=tk.BOTH, expand=True)
        self.results_list.bind("<<ListboxSelect>>", self._on_result_selected)
        panes.add(left, weight=1)

        right = ttk.Frame(panes)
        self.results_text = scrolledtext.ScrolledText(right, wrap=tk.WORD, font=("Arial", 11),
                                                      relief="solid", borderwidth=1)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)
        panes.add(right, weight=3)

        self._result_order: List[str] = []

    def _create_status_bar(self) -> None:
        bar = ttk.Frame(self.root, relief="sunken", padding=(5, 2))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Pronto")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(bar, text="Tarefa:").pack(side=tk.LEFT, padx=(10, 2))
        self.task_progress_bar = ttk.Progressbar(bar, orient="horizontal", length=150,
                                                 mode="determinate")
        self.task_progress_bar.pack(side=tk.LEFT, padx=5)
        ttk.Label(bar, text="Geral:").pack(side=tk.LEFT, padx=(10, 2))
        self.total_progress_bar = ttk.Progressbar(bar, orient="horizontal", length=150,
                                                  mode="determinate")
        self.total_progress_bar.pack(side=tk.LEFT, padx=5)

    # -- navegação -----------------------------------------------------------

    def show_main_screen(self) -> None:
        self.results_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)

    def show_results_screen(self) -> None:
        self.main_frame.pack_forget()
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    def _show_gpu_help(self) -> None:
        messagebox.showinfo(
            "Aceleração por GPU",
            f"Motivo: {self.device_info.problem or 'desconhecido'}\n\n"
            "Requisitos para usar a GPU:\n"
            "  1. Placa NVIDIA com driver atualizado.\n"
            "  2. Bibliotecas CUDA, instaláveis via pip:\n\n"
            "     pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n\n"
            "Não é necessário instalar o CUDA Toolkit nem o PyTorch."
        )

    # -- gestão da lista de tarefas -----------------------------------------

    def _add_tasks(self, sources: List[Any]) -> None:
        for source in sources:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
            display = source.name if isinstance(source, Path) else str(source)
            self.tasks[task_id] = Task(source=source, display=display)
            self.task_tree.insert("", tk.END, iid=task_id, values=(display, "Na Fila"))

    def _handle_drop(self, event) -> None:
        raw = event.data
        paths = re.findall(r"\{([^}]+)\}", raw) if "{" in raw else raw.split()
        if paths:
            self._add_tasks([Path(p) for p in paths])

    def _browse_files(self) -> None:
        filetypes = (
            ("Ficheiros Suportados",
             "*.mp3 *.mp4 *.m4a *.wav *.avi *.mkv *.mov *.flac *.ogg *.opus *.webm *.aac *.wma"),
            ("Todos os ficheiros", "*.*"),
        )
        paths = filedialog.askopenfilenames(title="Selecione ficheiros", filetypes=filetypes)
        if paths:
            self._add_tasks([Path(p) for p in paths])

    def _add_urls_dialog(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Adicionar URLs")
        dialog.geometry("560x340")
        dialog.transient(self.root)
        ttk.Label(dialog, text="Cole as URLs abaixo (uma por linha).\n"
                               "Qualquer site suportado pelo yt-dlp funciona, não só YouTube."
                  ).pack(pady=10, padx=10, anchor="w")
        url_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=10)
        url_text.pack(fill=tk.BOTH, expand=True, padx=10)
        url_text.focus_set()

        def on_add() -> None:
            lines = [u.strip() for u in url_text.get(1.0, tk.END).splitlines() if u.strip()]
            valid = [u for u in lines if URL_PATTERN.match(u)]
            invalid = [u for u in lines if u not in valid]
            dialog.destroy()
            if invalid:
                messagebox.showwarning(
                    "URLs ignoradas",
                    "Estas linhas não parecem URLs e foram ignoradas:\n\n" + "\n".join(invalid[:10])
                )
            if not valid:
                return
            playlists = [u for u in valid if "list=" in u]
            if playlists and messagebox.askyesno(
                "Playlist detectada",
                f"Detectei {len(playlists)} URL(s) de playlist.\n\n"
                "Deseja expandir em tarefas individuais?\n"
                "(Se escolher Não, apenas o vídeo referenciado será baixado.)"
            ):
                self._update_status("Lendo playlist(s)...")
                threading.Thread(target=self._expand_playlists_task,
                                 args=(valid, playlists), daemon=True).start()
                return
            self._add_tasks(valid)

        ttk.Button(dialog, text="Adicionar à Lista", command=on_add).pack(pady=10)

    def _expand_playlists_task(self, all_urls: List[str], playlists: List[str]) -> None:
        resolved: List[str] = []
        for url in all_urls:
            if url in playlists:
                try:
                    entries = expand_playlist(url)
                    resolved.extend(entries or [url])
                except Exception as exc:
                    self.task_queue.put(("status", f"Falha ao ler playlist: "
                                                   f"{friendly_download_error(exc)}"))
                    resolved.append(url)
            else:
                resolved.append(url)
        self.task_queue.put(("add_tasks", resolved))

    def _remove_selected(self) -> None:
        for item in self.task_tree.selection():
            self.task_tree.delete(item)
            self.tasks.pop(item, None)

    def _clear_task_list(self) -> None:
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        self.tasks.clear()
        self.task_counter = 0

    def _select_output_directory(self) -> None:
        path = filedialog.askdirectory(title="Selecione a pasta de destino")
        if path:
            self.output_directory = Path(path)
            self.output_dir_var.set(str(self.output_directory))

    def _open_output_dir(self) -> None:
        if not self.output_directory.is_dir():
            messagebox.showinfo("Sem pasta", "Nenhuma pasta de destino foi selecionada.")
            return
        if sys.platform == "win32":
            os.startfile(self.output_directory)  # noqa: S606 - explorer do Windows
        else:
            subprocess.Popen(["xdg-open" if sys.platform.startswith("linux") else "open",
                              str(self.output_directory)])

    # -- estimativa de VRAM --------------------------------------------------

    def _effective_compute_type(self) -> str:
        chosen = self.compute_var.get()
        if chosen != "Automático":
            return chosen
        return auto_compute_type(self.model_var.get(), self.device_info)

    def _refresh_estimate(self) -> None:
        if self.backend_var.get() != "faster-whisper":
            self.estimate_label.config(
                text="openai-whisper: consome ~2x mais memória", style="Warn.TLabel")
            return
        compute = self._effective_compute_type()
        if not self.device_info.is_cuda:
            self.estimate_label.config(text=f"CPU · precisão {compute}", style="TLabel")
            return
        needed = vram_estimate_mb(self.model_var.get(), compute)
        budget = self.device_info.usable_vram_mb
        if budget and needed > budget:
            self.estimate_label.config(
                text=f"⚠ ~{needed} MiB > {budget} MiB livres ({compute})", style="Warn.TLabel")
        else:
            self.estimate_label.config(
                text=f"✓ ~{needed} MiB de VRAM ({compute})", style="Ok.TLabel")

    # -- processamento -------------------------------------------------------

    def _selected_formats(self) -> List[str]:
        return [fmt for fmt, var in self.format_vars.items() if var.get()]

    def _start_processing(self) -> None:
        if self.is_processing:
            return
        task_ids = list(self.task_tree.get_children())
        if not task_ids:
            messagebox.showwarning("Nenhuma Tarefa",
                                   "Adicione ficheiros ou URLs à lista de tarefas.")
            return
        if not self._selected_formats():
            messagebox.showwarning("Nenhum Formato", "Selecione ao menos um formato de saída.")
            return

        tasks = [self.tasks[tid] for tid in task_ids]
        needs_folder = len(tasks) > 1 or any(t.is_url for t in tasks)
        if needs_folder and not self.output_directory.is_dir():
            messagebox.showerror("Pasta de Destino Necessária",
                                 "Para processar múltiplos itens ou URLs, "
                                 "selecione uma pasta de destino.")
            return

        # O backend legado usa o binário do FFmpeg para decodificar; o faster-whisper não.
        if self.backend_var.get() == "openai-whisper" and not self.ffmpeg_path.is_file():
            messagebox.showerror(
                "FFmpeg Necessário",
                "O backend 'openai-whisper' precisa do FFmpeg instalado e no PATH.\n\n"
                "Use o backend 'faster-whisper' (padrão), que decodifica via PyAV e "
                "dispensa o FFmpeg.")
            return

        if self.device_info.is_cuda and self.backend_var.get() == "faster-whisper":
            compute = self._effective_compute_type()
            needed = vram_estimate_mb(self.model_var.get(), compute)
            budget = self.device_info.usable_vram_mb
            if budget and needed > budget and not messagebox.askyesno(
                "VRAM insuficiente",
                f"O modelo '{self.model_var.get()}' em {compute} precisa de ~{needed} MiB, "
                f"mas só há ~{budget} MiB livres.\n\n"
                "A transcrição vai transbordar para a memória do sistema e ficar MUITO lenta.\n\n"
                "Sugestão: use 'large-v3-turbo' com precisão 'int8_float16'.\n\n"
                "Deseja continuar mesmo assim?"
            ):
                return

        self._persist_settings()
        self.is_processing = True
        self.cancel_event.clear()
        self._toggle_controls(enabled=False)
        self.cancel_button.config(state="normal")
        self.total_progress_bar["value"] = 0
        threading.Thread(target=self._process_batch_task, args=(task_ids,), daemon=True).start()

    def _cancel_processing(self) -> None:
        if self.is_processing:
            self.cancel_event.set()
            self._update_status("Cancelando... (a tarefa atual será interrompida)")
            self.cancel_button.config(state="disabled")

    def _process_batch_task(self, task_ids: List[str]) -> None:
        temp_dir = Path(tempfile.gettempdir()) / "audio_transcriber"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            backend = (FasterWhisperBackend() if self.backend_var.get() == "faster-whisper"
                       else OpenAIWhisperBackend())
            if type(backend) is not type(self.backend):
                self.backend.unload()
                self.backend = backend

            model_name = self.model_var.get()
            compute_type = self._effective_compute_type()
            language = self.LANGUAGES[self.lang_var.get()]
            task_type = "translate" if self.translate_var.get() else "transcribe"
            use_vad = self.vad_var.get()
            formats = self._selected_formats()
            cookies = self.cookies_var.get()
            save_to_disk = self.output_directory.is_dir()

            self.task_queue.put(("status", f"Carregando modelo '{model_name}' "
                                           f"({self.device_info.device}/{compute_type})..."))
            self.backend.load(model_name, self.device_info.device, compute_type)

            total = len(task_ids)
            for index, task_id in enumerate(task_ids):
                if self.cancel_event.is_set():
                    self._mark_remaining_cancelled(task_ids[index:])
                    break
                task = self.tasks[task_id]
                self.task_queue.put(("total_progress", index / total * 100))
                self.task_queue.put(("task_progress", 0))
                try:
                    self._process_single_task(task_id, task, temp_dir, cookies, language,
                                              task_type, use_vad, formats, save_to_disk)
                except Cancelled:
                    self._mark_remaining_cancelled(task_ids[index:])
                    break

            self.task_queue.put(("total_progress", 100))
            self.task_queue.put(("results_ready", None))
        except Exception as exc:
            self.task_queue.put(("error", f"Falha no processamento em lote: {exc}"))
            traceback.print_exc()
        finally:
            self.task_queue.put(("finish", None))

    def _process_single_task(self, task_id: str, task: Task, temp_dir: Path, cookies: str,
                             language: Optional[str], task_type: str, use_vad: bool,
                             formats: List[str], save_to_disk: bool) -> None:
        media_path: Optional[Path] = None
        temp_file: Optional[Path] = None
        output_stem = Path(str(task.source)).stem

        if task.is_url:
            self.task_queue.put(("update_status", (task_id, "Baixando...")))
            try:
                opts = build_ydl_opts(temp_dir, cookies, self._yt_dlp_hook)
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(task.source, download=True)
                    media_path = downloaded_filepath(ydl, info)
                    temp_file = media_path
                    output_stem = safe_output_stem(info.get("title", ""), media_path.stem)
            except Exception as exc:
                # O hook de progresso levanta Cancelled, que o yt-dlp reembrulha.
                if self.cancel_event.is_set():
                    raise Cancelled() from exc
                self._fail_task(task_id, task, "Erro no Download", friendly_download_error(exc))
                return
        else:
            media_path = Path(task.source)
            if not media_path.is_file():
                self._fail_task(task_id, task, "Ficheiro não encontrado",
                                f"O ficheiro '{media_path}' não existe.")
                return

        try:
            self.task_queue.put(("update_status", (task_id, "Transcrevendo...")))
            result = self.backend.transcribe(
                media_path, language, task_type, use_vad,
                lambda pct: self.task_queue.put(("task_progress", pct)),
                self.cancel_event,
            )
            task.text = result.text
            if save_to_disk:
                task.saved = self._write_outputs(output_stem, result, formats)
                label = f"OK · {', '.join('.' + f for f in formats)}"
            else:
                label = "Concluído"
            task.status = label
            self.task_queue.put(("update_status", (task_id, label)))
        except Cancelled:
            task.status = "Cancelado"
            self.task_queue.put(("update_status", (task_id, "Cancelado")))
            raise
        except Exception as exc:
            self._fail_task(task_id, task, "Erro na Transcrição", str(exc))
            traceback.print_exc()
        finally:
            if temp_file is not None and temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError as exc:
                    print(f"Erro ao limpar ficheiro temporário {temp_file}: {exc}")

    def _write_outputs(self, stem: str, result: TranscriptionResult,
                       formats: List[str]) -> List[Path]:
        written: List[Path] = []
        for fmt in formats:
            path = self.output_directory / f"{stem}.{fmt}"
            content = WRITERS[fmt](result.segments, result.text)
            path.write_text(content, encoding="utf-8")
            written.append(path)
        return written

    def _fail_task(self, task_id: str, task: Task, status: str, detail: str) -> None:
        task.status = status
        task.error = detail
        self.task_queue.put(("update_status", (task_id, status)))

    def _mark_remaining_cancelled(self, task_ids: List[str]) -> None:
        for task_id in task_ids:
            task = self.tasks.get(task_id)
            if task and task.status in ("Na Fila", "Processando..."):
                task.status = "Cancelado"
                self.task_queue.put(("update_status", (task_id, "Cancelado")))

    def _yt_dlp_hook(self, data: Dict[str, Any]) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            if total:
                self.task_queue.put(("task_progress", data.get("downloaded_bytes", 0) / total * 100))

    # -- bomba de eventos ----------------------------------------------------

    def _process_queue(self) -> None:
        try:
            while not self.task_queue.empty():
                message, data = self.task_queue.get_nowait()
                if message == "status":
                    self._update_status(data)
                elif message == "task_progress":
                    self.task_progress_bar["value"] = data
                elif message == "total_progress":
                    self.total_progress_bar["value"] = data
                elif message == "update_status":
                    task_id, text = data
                    if self.task_tree.exists(task_id):
                        self.task_tree.set(task_id, "status", text)
                elif message == "add_tasks":
                    self._add_tasks(data)
                    self._update_status(f"{len(data)} tarefa(s) adicionada(s).")
                elif message == "results_ready":
                    self._display_results()
                elif message == "info":
                    messagebox.showinfo("Informação", data)
                elif message == "error":
                    self._handle_error(data)
                elif message == "finish":
                    self.is_processing = False
                    self._toggle_controls(enabled=True)
                    self.cancel_button.config(state="disabled")
                    if self.cancel_event.is_set():
                        self._update_status("Processamento cancelado.")
                    else:
                        self._update_status("Processo concluído!")
        finally:
            self.root.after(100, self._process_queue)

    def _toggle_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.controls_to_toggle:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass

    # -- resultados ----------------------------------------------------------

    def _display_results(self) -> None:
        self.results_list.delete(0, tk.END)
        self._result_order = []
        for task_id, task in self.tasks.items():
            self._result_order.append(task_id)
            marker = "✓" if task.text else "✗"
            self.results_list.insert(tk.END, f"{marker} {task.display[:60]}")
        if self._result_order:
            self.results_list.selection_set(0)
            self._render_result(self._result_order[0])
        self.show_results_screen()

    def _on_result_selected(self, _event) -> None:
        selection = self.results_list.curselection()
        if selection:
            self._render_result(self._result_order[selection[0]])

    def _render_result(self, task_id: str) -> None:
        task = self.tasks[task_id]
        lines = [f"--- {task.display} ---", f"Status: {task.status}"]
        if task.saved:
            lines.append("Salvo em: " + ", ".join(p.name for p in task.saved))
        lines.append("")
        lines.append(task.error if task.error else (task.text or "(sem texto)"))
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.config(state=tk.DISABLED)

    def _save_current(self) -> None:
        content = self.results_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("Vazio", "Não há nada para salvar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Texto", "*.txt"), ("Legenda SRT", "*.srt"), ("Todos", "*.*")],
            title="Salvar Transcrição")
        if not path:
            return
        try:
            Path(path).write_text(content, encoding="utf-8")
            self._update_status(f"Salvo em {path}")
        except OSError as exc:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar:\n{exc}")

    def _copy_to_clipboard(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(self.results_text.get(1.0, tk.END))
        self._update_status("Texto copiado para a área de transferência.")

    # -- utilitários ---------------------------------------------------------

    def _handle_error(self, message: str) -> None:
        self.is_processing = False
        self._update_status("Ocorreu um erro.")
        self.total_progress_bar["value"] = 0
        self.task_progress_bar["value"] = 0
        messagebox.showerror("Erro", str(message))
        self._toggle_controls(enabled=True)

    def _update_status(self, message: str) -> None:
        self.status_var.set(message)

    def _persist_settings(self) -> None:
        save_settings({
            "model": self.model_var.get(),
            "compute_type": self.compute_var.get(),
            "language": self.lang_var.get(),
            "vad": self.vad_var.get(),
            "cookies": self.cookies_var.get(),
            "formats": self._selected_formats(),
            "output_dir": str(self.output_directory) if self.output_directory.is_dir() else "",
        })

    def _on_close(self) -> None:
        if self.is_processing and not messagebox.askyesno(
            "Processamento em andamento", "Há uma transcrição em curso. Deseja sair mesmo assim?"
        ):
            return
        self.cancel_event.set()
        self._persist_settings()
        self.backend.unload()
        self.root.destroy()

    # -- FFmpeg (opcional) ---------------------------------------------------
    #
    # O FFmpeg deixou de ser obrigatório: o PyAV (dependência do faster-whisper)
    # decodifica os formatos comuns. Mantido para o backend openai-whisper e para
    # containers exóticos.

    def _find_ffmpeg_path(self) -> Path:
        exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        local = Path(__file__).parent.resolve() / "ffmpeg" / exe
        if local.is_file():
            return local
        system = shutil.which("ffmpeg")
        return Path(system) if system else Path()

    def _setup_ffmpeg_environment(self) -> None:
        self.ffmpeg_path = self._find_ffmpeg_path()
        if not self.ffmpeg_path.is_file():
            return
        ffmpeg_dir = str(self.ffmpeg_path.parent)
        current = os.environ.get("PATH", "")
        parts = [p for p in current.split(os.pathsep) if p]
        if not any(os.path.normcase(p) == os.path.normcase(ffmpeg_dir) for p in parts):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + current


def main() -> None:
    # A raiz precisa ser TkinterDND2 para o drag-and-drop funcionar.
    root = tkinterdnd2.Tk()
    AudioTranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
