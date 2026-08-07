# coding: utf-8
"""
Aplicativo de Transcrição de Áudio e Vídeo (Versão Final e Robusta)

Ferramenta focada em transcrever ficheiros de áudio/vídeo locais ou do YouTube
usando a tecnologia Whisper da OpenAI.

Funcionalidades:
- Interface baseada em lista de tarefas para clareza no processamento em lote.
- Tratamento de erros de rede: o processo continua para o próximo item em caso de falha.
- Transcrição em lote de múltiplos ficheiros locais e URLs do YouTube.
- Salvamento automático em pasta definida para trabalhos em lote.
- Feedback de progresso detalhado com barras para tarefa atual e progresso geral.
- Opção de otimização para GPU (fp16) para maior performance.
- Gestão segura de ficheiros temporários e dependências.
- Download e configuração automática (opcional) do FFmpeg.
"""

import sys
import json
import queue
import subprocess
import tempfile
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
import urllib.request
import zipfile
import tarfile
import shutil
import os

# --- Bloco de Verificação e Instalação de Dependências ---

def check_and_install_dependencies():
    """Verifica as dependências apenas uma vez, usando um ficheiro de flag."""
    flag_file = Path.home() / ".transcriber_deps_ok"
    if flag_file.exists():
        return True

    try:
        import torch
        import whisper
        import yt_dlp
        import tkinterdnd2
        flag_file.touch()
        return True
    except ImportError as e:
        missing_module = e.name
        root = tk.Tk()
        root.withdraw()
        packages = ["openai-whisper", "torch", "yt-dlp", "tkinterdnd2"]
        response = messagebox.askyesno(
            "Dependências em Falta",
            f"O módulo '{missing_module}' é necessário.\nDeseja instalar as dependências agora?\n({', '.join(packages)})"
        )
        if response:
            try:
                command = [sys.executable, "-m", "pip", "install"] + packages
                subprocess.run(command, check=True, capture_output=True, text=True)
                flag_file.touch()
                messagebox.showinfo("Instalação Concluída", "Dependências instaladas. Por favor, reinicie o aplicativo.")
                sys.exit(0)
            except Exception as install_error:
                messagebox.showerror("Erro na Instalação", f"Falha ao instalar: {install_error}\n\nInstale manualmente: pip install {' '.join(packages)}")
                sys.exit(1)
        else:
            sys.exit(1)

check_and_install_dependencies()

import whisper
import yt_dlp
import tkinterdnd2

# --- Classe Principal da Aplicação ---

class AudioTranscriberApp:
    LANGUAGES = {
        "Detectar Automaticamente": None, "Português": "pt", "Inglês": "en", 
        "Espanhol": "es", "Francês": "fr", "Alemão": "de", "Italiano": "it",
        "Japonês": "ja", "Chinês": "zh", "Russo": "ru"
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Transcrição de Áudio e Vídeo com Whisper")
        self.root.minsize(800, 700)
        self.root.geometry("1024x768")

        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.style.configure("Accent.TButton", font=("Arial", 12, "bold"))
        self.style.configure("Treeview.Heading", font=('Arial', 10, 'bold'))

        # Variáveis de estado
        self.output_directory: Path = Path()
        self.whisper_model_cache: Dict[str, Any] = {}
        self.task_queue = queue.Queue()
        self.ffmpeg_path: Path = Path()
        self.is_processing = False
        self.task_map: Dict[str, Any] = {}
        self.task_counter = 0
        
        self._setup_ffmpeg_environment()

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.results_frame = ttk.Frame(self.root, padding="10")
        
        self.controls_to_toggle = []
        self._create_main_widgets()
        self._create_results_widgets()
        self._create_status_bar()

        self.root.after(100, self._process_queue)
        self.show_main_screen()

    def _create_main_widgets(self):
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        task_frame = ttk.LabelFrame(self.main_frame, text="1. Lista de Tarefas de Transcrição", padding="10")
        task_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        task_buttons_frame = ttk.Frame(task_frame)
        task_buttons_frame.pack(fill=tk.X, pady=(0, 5))
        
        btn_add_files = ttk.Button(task_buttons_frame, text="Adicionar Ficheiros...", command=self._browse_files)
        btn_add_files.pack(side=tk.LEFT)
        
        btn_add_urls = ttk.Button(task_buttons_frame, text="Adicionar URLs do YouTube...", command=self._add_youtube_urls_dialog)
        btn_add_urls.pack(side=tk.LEFT, padx=10)
        
        btn_clear = ttk.Button(task_buttons_frame, text="Limpar Lista", command=self._clear_task_list)
        btn_clear.pack(side=tk.RIGHT)
        
        self.controls_to_toggle.extend([btn_add_files, btn_add_urls, btn_clear])

        self.task_tree = ttk.Treeview(task_frame, columns=("source", "status"), show="headings", height=8)
        self.task_tree.heading("source", text="Ficheiro ou URL")
        self.task_tree.heading("status", text="Status")
        self.task_tree.column("source", width=600)
        self.task_tree.column("status", width=150, anchor="center")
        self.task_tree.pack(fill=tk.BOTH, expand=True)
        self.task_tree.drop_target_register('DND_Files')
        self.task_tree.dnd_bind('<<Drop>>', self._handle_drop)

        config_frame = ttk.LabelFrame(self.main_frame, text="2. Configurações da Transcrição", padding="10")
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill=tk.X)
        
        ttk.Label(config_grid, text="Modelo Whisper:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_var = tk.StringVar(value="base")
        model_combo = ttk.Combobox(config_grid, textvariable=self.model_var, values=["tiny", "base", "small", "medium", "large"], state="readonly")
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        ttk.Label(config_grid, text="Idioma do Áudio:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.lang_var = tk.StringVar(value="Detectar Automaticamente")
        lang_combo = ttk.Combobox(config_grid, textvariable=self.lang_var, values=list(self.LANGUAGES.keys()), state="readonly")
        lang_combo.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        self.translate_var = tk.BooleanVar()
        translate_check = ttk.Checkbutton(config_grid, text="Traduzir para Inglês", variable=self.translate_var)
        translate_check.grid(row=1, column=2, padx=10, pady=5, sticky="w")
        
        self.fp16_var = tk.BooleanVar(value=False)
        fp16_check = ttk.Checkbutton(config_grid, text="Otimização para GPU (fp16)", variable=self.fp16_var)
        fp16_check.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        config_grid.columnconfigure(1, weight=1)
        self.controls_to_toggle.extend([model_combo, lang_combo, translate_check, fp16_check])

        output_frame = ttk.LabelFrame(self.main_frame, text="3. Pasta de Destino (Obrigatório para salvar em lote)", padding="10")
        output_frame.pack(fill=tk.X, padx=10, pady=5)
        self.output_dir_var = tk.StringVar(value="Nenhuma pasta selecionada.")
        output_dir_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, state="readonly")
        output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        btn_select_folder = ttk.Button(output_frame, text="Selecionar Pasta...", command=self._select_output_directory)
        btn_select_folder.pack(side=tk.LEFT)
        self.controls_to_toggle.append(btn_select_folder)

        action_frame = ttk.Frame(self.main_frame)
        action_frame.pack(fill=tk.X, padx=10, pady=15)
        self.start_button = ttk.Button(action_frame, text="INICIAR PROCESSAMENTO", command=self._start_processing, style="Accent.TButton")
        self.start_button.pack(side=tk.RIGHT, ipady=10, ipadx=10)
        self.controls_to_toggle.append(self.start_button)

    def _create_results_widgets(self):
        toolbar = ttk.Frame(self.results_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(toolbar, text="Salvar Transcrição (.txt)", command=self._save_transcription).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Copiar Tudo", command=self._copy_to_clipboard).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="← Voltar", command=self.show_main_screen).pack(side=tk.RIGHT)
        self.results_text = scrolledtext.ScrolledText(self.results_frame, wrap=tk.WORD, font=("Arial", 11), relief="solid", borderwidth=1)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)

    def _create_status_bar(self):
        status_frame = ttk.Frame(self.root, relief="sunken", padding=(5, 2))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Pronto")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.task_progress_label = ttk.Label(status_frame, text="Tarefa Atual:")
        self.task_progress_label.pack(side=tk.LEFT, padx=(10, 2))
        self.task_progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=150, mode="determinate")
        self.task_progress_bar.pack(side=tk.LEFT, padx=5)
        self.total_progress_label = ttk.Label(status_frame, text="Progresso Geral:")
        self.total_progress_label.pack(side=tk.LEFT, padx=(10, 2))
        self.total_progress_bar = ttk.Progressbar(status_frame, orient="horizontal", length=150, mode="determinate")
        self.total_progress_bar.pack(side=tk.LEFT, padx=5)
    
    def show_main_screen(self):
        self.results_frame.pack_forget()
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._reset_state()

    def show_results_screen(self):
        self.main_frame.pack_forget()
        self.results_frame.pack(fill=tk.BOTH, expand=True)

    def _reset_state(self):
        self._clear_task_list()
        self._update_status("Pronto")
        self.total_progress_bar['value'] = 0
        self.task_progress_bar['value'] = 0

    def _add_tasks_to_list(self, tasks: List):
        for task_source in tasks:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}"
            self.task_map[task_id] = task_source
            
            source_display = task_source.name if isinstance(task_source, Path) else str(task_source)
            if not self.task_tree.exists(task_id):
                self.task_tree.insert("", tk.END, iid=task_id, values=(source_display, "Na Fila"))

    def _handle_drop(self, event):
        filepaths_str = event.data
        if '{' in filepaths_str:
            filepaths = re.findall(r'\{([^}]+)\}', filepaths_str)
        else:
            filepaths = filepaths_str.split()
        if filepaths:
            self._add_tasks_to_list([Path(fp) for fp in filepaths])

    def _browse_files(self):
        filepaths = filedialog.askopenfilenames(title="Selecione um ou mais ficheiros", filetypes=(("Ficheiros Suportados", "*.mp3 *.mp4 *.m4a *.wav *.avi *.mkv *.mov"), ("Todos os ficheiros", "*.*")))
        if filepaths:
            self._add_tasks_to_list([Path(fp) for fp in filepaths])

    def _add_youtube_urls_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Adicionar URLs do YouTube")
        dialog.geometry("500x300")
        ttk.Label(dialog, text="Cole as URLs abaixo (uma por linha):").pack(pady=10, padx=10, anchor="w")
        url_text = scrolledtext.ScrolledText(dialog, wrap=tk.WORD, height=10)
        url_text.pack(fill=tk.BOTH, expand=True, padx=10)
        url_text.focus_set()
        def on_add():
            urls = [url.strip() for url in url_text.get(1.0, tk.END).strip().splitlines() if url.strip()]
            valid_urls = [url for url in urls if "youtube.com" in url or "youtu.be" in url]
            self._add_tasks_to_list(valid_urls)
            dialog.destroy()
        ttk.Button(dialog, text="Adicionar à Lista", command=on_add).pack(pady=10)

    def _clear_task_list(self):
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        self.task_map.clear()
        self.task_counter = 0

    def _select_output_directory(self):
        dir_path = filedialog.askdirectory(title="Selecione a pasta para salvar as transcrições")
        if dir_path:
            self.output_directory = Path(dir_path)
            self.output_dir_var.set(str(self.output_directory))

    def _start_processing(self):
        if self.is_processing:
            return
            
        try:
            tasks = self.task_tree.get_children()
            if not tasks:
                messagebox.showwarning("Nenhuma Tarefa", "Por favor, adicione ficheiros ou URLs à lista de tarefas.")
                return

            task_sources = [self.task_map[task_id] for task_id in tasks]
            is_batch = len(tasks) > 1
            is_url_task = any(isinstance(src, str) and src.startswith("http") for src in task_sources)
            
            auto_save_required = is_batch or is_url_task
            
            if auto_save_required and not self.output_directory.is_dir():
                messagebox.showerror("Pasta de Destino Necessária", "Para processar múltiplos itens ou URLs do YouTube, você deve selecionar uma pasta de destino.")
                return

            has_video_or_url = any(not str(src).lower().endswith(('.wav', '.mp3')) for src in task_sources)
            if has_video_or_url and not self._check_ffmpeg():
                should_download = messagebox.askyesno("FFmpeg Não Encontrado", "Para processar vídeos ou URLs, o FFmpeg é necessário.\nDeseja que o aplicativo tente baixar e configurar o FFmpeg automaticamente?")
                if should_download:
                    self.is_processing = True
                    self._update_status("Iniciando download do FFmpeg...")
                    self._toggle_controls(enabled=False)
                    self._run_task_in_thread(self._ffmpeg_download_task)
                return

            self.is_processing = True
            self._toggle_controls(enabled=False)
            self.total_progress_bar['value'] = 0
            auto_save_enabled = self.output_directory.is_dir()
            self._run_task_in_thread(self._process_batch_task, tasks, auto_save_enabled)

        except Exception as e:
            self._handle_error(f"Ocorreu um erro inesperado ao iniciar o processo: {e}")

    def _run_task_in_thread(self, target_func, *args):
        thread = threading.Thread(target=target_func, args=args, daemon=True)
        thread.start()

    def _process_batch_task(self, tasks: List[str], auto_save: bool):
        try:
            model_name = self.model_var.get()
            language_code = self.LANGUAGES[self.lang_var.get()]
            transcribe_task_type = "translate" if self.translate_var.get() else "transcribe"
            use_fp16 = self.fp16_var.get()
            
            self.task_queue.put(("status", f"Carregando modelo Whisper '{model_name}'..."))
            if model_name not in self.whisper_model_cache:
                self.whisper_model_cache[model_name] = whisper.load_model(model_name)
            model = self.whisper_model_cache[model_name]
            
            results_log = []
            temp_dir = Path(tempfile.gettempdir()) / "audio_transcriber"
            temp_dir.mkdir(exist_ok=True)

            for i, task_id in enumerate(tasks):
                self.task_queue.put(("total_progress", (i / len(tasks)) * 100))
                self.task_queue.put(("update_task_status", (task_id, "Processando...")))
                
                task_source = self.task_map[task_id]
                file_to_transcribe = None
                original_name = Path(str(task_source)).stem
                is_temp_file = False

                if isinstance(task_source, str) and task_source.startswith("http"):
                    try:
                        self.task_queue.put(("update_task_status", (task_id, "Baixando...")))
                        ydl_opts = {'format': 'bestaudio/best', 'outtmpl': str(temp_dir / '%(title)s.%(ext)s'), 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}], 'quiet': True, 'no_warnings': True, 'progress_hooks': [self._yt_dlp_hook]}
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(task_source, download=True)
                            file_to_transcribe = Path(ydl.prepare_filename(info)).with_suffix('.wav')
                            original_name = file_to_transcribe.stem
                            is_temp_file = True
                    except Exception as e:
                        results_log.append(f"FALHA ao baixar '{task_source}': {e}")
                        self.task_queue.put(("update_task_status", (task_id, "Erro de Rede")))
                        continue
                else:
                    file_to_transcribe = Path(task_source)

                try:
                    self.task_queue.put(("update_task_status", (task_id, "Transcrevendo...")))
                    self.task_queue.put(("task_progress_mode", "indeterminate"))
                    
                    result = model.transcribe(str(file_to_transcribe), language=language_code, task=transcribe_task_type, fp16=use_fp16)
                    transcription_text = result["text"].strip()
                    
                    self.task_queue.put(("task_progress_mode", "determinate"))
                    self.task_queue.put(("task_progress", 100))

                    if auto_save:
                        output_filename = f"{original_name}.txt"
                        output_path = self.output_directory / output_filename
                        try:
                            with open(output_path, 'w', encoding='utf-8') as f: f.write(transcription_text)
                            results_log.append(f"SUCESSO: '{original_name}.txt' salvo em '{self.output_directory.name}'")
                            self.task_queue.put(("update_task_status", (task_id, "Concluído")))
                        except Exception as e:
                            results_log.append(f"FALHA ao salvar '{original_name}.txt': {e}")
                            self.task_queue.put(("update_task_status", (task_id, "Erro ao Salvar")))
                    else:
                        results_log.append((original_name, transcription_text))
                        self.task_queue.put(("update_task_status", (task_id, "Concluído")))
                except Exception as e:
                    results_log.append(f"FALHA na transcrição de '{original_name}': {e}")
                    self.task_queue.put(("update_task_status", (task_id, "Erro na Transcrição")))
                finally:
                    if is_temp_file and file_to_transcribe.exists():
                        try:
                            file_to_transcribe.unlink()
                        except OSError as e:
                            print(f"Erro ao limpar ficheiro temporário {file_to_transcribe}: {e}")

            self.task_queue.put(("total_progress", 100))
            self.task_queue.put(("result", results_log))
            self.task_queue.put(("finish", None))
        except Exception as e:
            self.task_queue.put(("error", f"Falha no processamento em lote: {e}"))
            self.task_queue.put(("finish", None))

    def _yt_dlp_hook(self, d):
        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total_bytes:
                progress = (d['downloaded_bytes'] / total_bytes) * 100
                self.task_queue.put(("task_progress", progress))

    def _ffmpeg_download_task(self):
        try:
            script_dir = Path(__file__).parent.resolve()
            ffmpeg_dir = script_dir / "ffmpeg"
            ffmpeg_dir.mkdir(exist_ok=True)
            if sys.platform == "win32":
                url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
                archive_path = ffmpeg_dir / "ffmpeg.zip"
            else:
                self.task_queue.put(("error", "Download automático de FFmpeg não suportado neste SO. Instale manualmente."))
                return
            
            self.task_queue.put(("status", "Baixando FFmpeg... (pode levar alguns minutos)"))
            urllib.request.urlretrieve(url, archive_path)
            self.task_queue.put(("total_progress", 50))
            self.task_queue.put(("status", "Extraindo FFmpeg..."))
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(ffmpeg_dir)
            archive_path.unlink()
            
            ffmpeg_exe_name = "ffmpeg.exe"
            try:
                found_exe = next(ffmpeg_dir.glob(f"**/{ffmpeg_exe_name}"))
                extracted_dir = found_exe.parent
                final_ffmpeg_path = ffmpeg_dir / ffmpeg_exe_name
                if found_exe != final_ffmpeg_path:
                    shutil.move(str(found_exe), str(final_ffmpeg_path))
                
                # Limpeza segura
                if extracted_dir != ffmpeg_dir:
                    shutil.rmtree(extracted_dir)

            except StopIteration:
                raise FileNotFoundError("ffmpeg.exe não encontrado no ficheiro baixado.")

            self._setup_ffmpeg_environment()
            self.task_queue.put(("total_progress", 100))
            self.task_queue.put(("status", "FFmpeg configurado! Por favor, inicie a transcrição novamente."))
            messagebox.showinfo("Sucesso", "FFmpeg foi baixado e configurado. Já pode iniciar a sua transcrição de vídeo.")
        except Exception as e:
            self.task_queue.put(("error", f"Falha ao baixar FFmpeg: {e}"))
        finally:
            self.task_queue.put(("finish", None))
            
    def _process_queue(self):
        try:
            while not self.task_queue.empty():
                message, data = self.task_queue.get_nowait()
                if message == "status": self._update_status(data)
                elif message == "task_progress": self.task_progress_bar['value'] = data
                elif message == "total_progress": self.total_progress_bar['value'] = data
                elif message == "task_progress_mode":
                    self.task_progress_bar['mode'] = data
                    if data == 'indeterminate': self.task_progress_bar.start()
                    else: self.task_progress_bar.stop()
                elif message == "update_task_status":
                    task_id, status_text = data
                    if self.task_tree.exists(task_id):
                        self.task_tree.set(task_id, "status", status_text)
                elif message == "result": self._display_results(data)
                elif message == "error": self._handle_error(data)
                elif message == "finish":
                    self.is_processing = False
                    self._toggle_controls(enabled=True)
                    self._update_status("Processo concluído!")
        finally:
            self.root.after(100, self._process_queue)

    def _toggle_controls(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for widget in self.controls_to_toggle:
            widget.config(state=state)

    def _display_results(self, results):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        if not results:
            self.results_text.insert(tk.END, "Nenhum resultado gerado.")
        elif isinstance(results[0], str):
            header = "--- RELATÓRIO DE TRANSCRIÇÃO EM LOTE ---\n\n"
            log_text = "\n".join(results)
            self.results_text.insert(tk.END, header + log_text)
        elif isinstance(results[0], tuple):
            filename, text = results[0]
            header = f"--- TRANSCRIÇÃO DE: {filename} ---\n\n"
            self.results_text.insert(tk.END, header + text)
        self.results_text.config(state=tk.DISABLED)
        self.show_results_screen()

    def _handle_error(self, error_message: str):
        self.is_processing = False
        self._update_status("Ocorreu um erro.")
        self.total_progress_bar['value'] = 0
        self.task_progress_bar['value'] = 0
        messagebox.showerror("Erro", str(error_message))
        traceback.print_exc()
        self._toggle_controls(enabled=True)

    def _update_status(self, message: str):
        self.status_var.set(message)
        
    def _find_ffmpeg_path(self) -> Path:
        ffmpeg_exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        script_dir = Path(__file__).parent.resolve()
        local_ffmpeg_dir = script_dir / "ffmpeg"
        local_ffmpeg_exe = local_ffmpeg_dir / ffmpeg_exe_name
        if local_ffmpeg_exe.is_file():
            return local_ffmpeg_exe
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            return Path(system_ffmpeg)
        return Path()

    def _setup_ffmpeg_environment(self):
        self.ffmpeg_path = self._find_ffmpeg_path()
        if self.ffmpeg_path.is_file():
            ffmpeg_dir = str(self.ffmpeg_path.parent)
            if ffmpeg_dir not in os.environ["PATH"]:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
                print(f"FFmpeg adicionado ao PATH da sessão: {ffmpeg_dir}")

    def _check_ffmpeg(self) -> bool:
        return self.ffmpeg_path.is_file()

    def _save_transcription(self):
        content = self.results_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("Vazio", "Não há nada para salvar.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Ficheiros de Texto", "*.txt"), ("Todos os Ficheiros", "*.*")], title="Salvar Transcrição")
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
                self._update_status(f"Transcrição salva em {filepath}")
            except IOError as e:
                messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o ficheiro:\n{e}")

    def _copy_to_clipboard(self):
        content = self.results_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self._update_status("Texto copiado para a área de transferência.")

def main():
    # A raiz precisa ser TkinterDND2 para o D&D funcionar
    root = tkinterdnd2.Tk()
    app = AudioTranscriberApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
