# 🎙️ Transcription (sem AI)

Aplicativo desktop para transcrever áudio e vídeo usando o [Whisper](https://github.com/openai/whisper) da OpenAI — tudo rodando **localmente**, sem enviar nada para APIs de terceiros.

Interface simples em Tkinter com suporte a lote de arquivos, URLs do YouTube e drag-and-drop.

## ✨ Funcionalidades

- 📂 **Processamento em lote** — adicione vários arquivos ou URLs de uma vez e acompanhe o progresso de cada tarefa numa lista.
- ▶️ **Arquivos locais e YouTube** — arraste arquivos de áudio/vídeo direto para a janela ou cole links do YouTube para baixar e transcrever.
- 🌍 **Múltiplos idiomas** — detecção automática ou seleção manual (Português, Inglês, Espanhol, Francês, Alemão, Italiano, Japonês, Chinês, Russo), com opção de tradução para inglês.
- ⚡ **Aceleração por GPU** — opção de otimização fp16 para quem tem placa de vídeo compatível.
- 🛡️ **Resiliente a falhas de rede** — se um item do lote falhar (ex: vídeo indisponível), o processamento continua para os próximos.
- 🔧 **Setup automático** — instala as dependências Python e baixa o FFmpeg sozinho na primeira execução, sem passos manuais.

## 🧰 Requisitos

- Python 3.9+
- Windows (o download automático do FFmpeg é feito só para Windows; em outros sistemas, instale o FFmpeg manualmente e garanta que está no `PATH`)

Não é necessário instalar nada antes — na primeira execução o próprio aplicativo detecta as dependências faltantes (`torch`, `openai-whisper`, `yt-dlp`, `tkinterdnd2`) e oferece para instalá-las via pip.

## 🚀 Como usar

```bash
python transcription_semAI.py
```

1. Na primeira execução, confirme a instalação das dependências quando solicitado e reabra o aplicativo.
2. Adicione arquivos (botão ou arraste-e-solte) e/ou URLs do YouTube à lista de tarefas.
3. Escolha o modelo do Whisper (`tiny` a `large` — modelos maiores são mais precisos, porém mais lentos) e o idioma.
4. Para lotes com mais de um item ou URLs do YouTube, selecione uma pasta de destino — as transcrições são salvas automaticamente em `.txt`.
5. Clique em **INICIAR PROCESSAMENTO** e acompanhe o progresso na barra de status.

Ao final, é possível copiar o resultado para a área de transferência ou salvar manualmente em um novo arquivo `.txt`.

## 📌 Observações

- Modelos maiores do Whisper (`medium`, `large`) exigem bastante RAM/VRAM — comece com `base` ou `small` se notar lentidão.
- Vídeos e URLs precisam do FFmpeg; se não for encontrado no sistema, o app oferece para baixá-lo e configurá-lo automaticamente.
