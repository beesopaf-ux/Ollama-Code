# ollamacode

A terminal coding assistant for local language models. It runs in your terminal (Termux on Android, or any Linux/macOS terminal), talks to a model on your own machine, and can create files and run shell commands when you approve them.

## Features

- **Local models:** works with [Ollama](https://ollama.com), or any OpenAI-compatible server such as llama.cpp's server or LM Studio.
- **Import `.gguf` files:** give a path to a `.gguf` file and ollamacode imports it into Ollama.
- **Agents:** save a model plus a system prompt under a name and switch between them.
- **Tools:** the model can create files and run bash commands. Every bash command asks for your y/n approval first.
- **Web search (optional):** `web_search` and `fetch_url` tools, using DuckDuckGo (no setup) or a self-hosted SearXNG server.
- **File attachments:** `/file <path>` sends a text file with your next message.
- **Copy code:** each code block in a reply gets a copy button.

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com), or any OpenAI-compatible server
- Python packages in `requirements.txt`

## Install Ollama

On Termux:

```bash
pkg install ollama
```

On Linux, macOS, or Windows, download the installer from [ollama.com/download](https://ollama.com/download).

ollamacode starts Ollama for you when it launches. You can turn this off in settings.

## Install ollamacode

```bash
git clone https://github.com/YOUR-USERNAME/ollamacode.git
cd ollamacode
pip install -r requirements.txt
python3 ollamacode.py
```

On Termux, install Python and git first with `pkg install python git`.

## Usage

Run `python3 ollamacode.py` from the folder you want the assistant to work in. Files it creates and commands it runs stay in that folder.

| Key or command | What it does |
| --- | --- |
| Ctrl+S | Open settings |
| Ctrl+L | Clear the chat |
| `/model` | Pick a model |
| `/file <path>` | Attach a text file to the next message |
| `/files` | List attachments (`/files clear` removes them) |
| `/clear` | Clear the chat |
| `/quit` | Exit |

In settings you can choose the backend, server URL, API key, model, context size, web search, and agents.

## Configuration

Settings are saved to `~/.config/ollamacode/config.json`. The file is created on first save and is never part of this repository.

## Notes

- Tool use needs a model that supports it, such as `qwen2.5-coder` or `llama3.1`. Small models can chat, but can't create files or run commands.
- Context size applies to the Ollama backend. Larger values need more memory.
- Web search uses DuckDuckGo by default. A self-hosted SearXNG server is also supported by setting its URL in settings.

## License

Add a license before publishing if you want others to reuse the code.
