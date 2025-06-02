# Ghostwriter UX Exploration CLI

## Motivation

This project was born from a desire to explore **new forms of flow-based interaction with artificially intelligent companion species**. As AI becomes more integrated into our daily lives—not just as tools, but as creative partners and companions—our interfaces must evolve beyond the traditional paradigms of command and response. We need new ways to **write, think, and create in collaboration with AI**, where the boundaries between human and machine creativity blur, and where the interface itself becomes a site of experimentation.

The **Ghostwriter UX Exploration CLI** is a terminal-based text editor that lets you write with the help of AI completions, inspired by the idea of "flow"—a seamless, conversational, and improvisational interaction with your digital companion. It is not just about efficiency, but about **discovering new modes of co-authorship, companionship, and creative emergence**.

---

## Culture & Philosophical Context

This project is informed by a broader cultural and philosophical conversation about AI, creativity, and the future of human-machine relations. Most of this came about while trying to find a better tool for writing by thinking of the LLM more as a collaborator rather than a question-answerer or task doer. Found some essays that are helpful in that context:

- [Remarks on AI from NZ by Neal Stephenson](https://nealstephenson.substack.com/p/remarks-on-ai-from-nz):  
  Stephenson reflects on the evolving relationship between humans and AI, and the need for new metaphors and practices for living with these "companion species."

- [AI, Heidegger, and Evangelion by Fakepixels](https://fakepixels.substack.com/p/ai-heidegger-and-evangelion):  
  This essay explores the existential and philosophical dimensions of AI, drawing on Heidegger and pop culture to ask what it means to dwell with intelligent machines.

- [Zero Cool and the Oraculator by Near Future Laboratory](https://nearfuturelaboratory.com/blog/2025/05/zero-cool-and-the-oraculator/):  
  A meditation on the shifting boundaries between human and machine agency, and the playful, experimental spirit needed to invent new forms of interaction.

- [Can AI Writing Be More Than A Gimmick](https://www.newyorker.com/books/under-review/can-ai-writing-be-more-than-a-gimmick):
  The essay collection *Searches: Selfhood in the Digital Age* by Vauhini Vara explores the intersection of artificial intelligence and personal writing. Vara, a novelist and tech journalist, initially experimented with large language models like ChatGPT to aid her writing, particularly during moments where she felt blocked or needed help expressing grief about her sister's death. Initially, she was wary of AI threatening her craft but eventually found its potential to elicit rather than supply text.

---

## Features

- **Flow-based AI writing**: Type as you normally would, and receive real-time AI completions that you can accept, ignore, or modify.
- **Visual cursor and completion**: The interface keeps the cursor and AI suggestion visually in sync, even with wrapped lines.
- **Multi-model support**: Works with OpenAI and local models (like LM Studio).
- **Terminal-native**: Designed for writers, hackers, and experimenters who love the command line.

---

## Why a Terminal App?

The terminal is a space of **immediacy, focus, and experimentation**. By building in the terminal, we invite users to play with the boundaries of text, code, and conversation—without the distractions of modern GUIs. It's a place to prototype new forms of interaction, and to **reclaim the joy of writing with machines**.

---
## Main Functionality 
The program creates a terminal-based text editor that provides real-time AI text completions as you type. It connects to either:

OpenAI's API (using models like GPT-4)
LM Studio (for local AI models)
Key Features
Multi-line Text Editor: Full cursor navigation with support for:

Arrow keys for movement
Home/End keys
Backspace/Delete
Enter for new lines
Soft text wrapping
AI Text Completion:

Automatically suggests completions after 600ms of inactivity
Shows completions in blue italic text
Use Tab to accept a completion
Use Shift+Up/Down to cycle through multiple completion options
Requests 3 completions at once for variety
Smart Completion Behavior:

Completions update as you type matching characters
Loads more completions when you reach the end of available options
Clears completions when you type non-matching text
Visual Interface: Uses the ratatui library to create a TUI with:

Main editor area with borders
Yellow separator line
Debug area showing cursor position and completion info
Visual cursor highlighting
Architecture
The program uses:

Async processing with tokio for non-blocking AI requests
Message passing with mpsc channels to handle completion results
Terminal UI with crossterm and ratatui
Command-line arguments via clap for configuration
The AI completions are designed to continue your text naturally without adding quotes, ellipses, or other formatting - just plain prose continuation


## Getting Started

1. **Clone the repo and build with Cargo** (requires Rust and a compatible AI backend):

    ```sh
    git clone https://github.com/yourusername/ghostwriter_streaming_cli.git
    cd ghostwriter_streaming_cli
    cargo build --release
    ```

2. **Run the CLI** with the desired options. For example:

    ```sh
    cargo run --release -- --endpoint openai --api-key YOUR_OPENAI_API_KEY --model gpt-3.5-turbo
    ```

    Or, for LM Studio or another local model:

    ```sh
    cargo run --release -- --endpoint http://localhost:1234/v1 --model your-model-name
    ```

    **Command-line options:**
    - `--endpoint` — The API endpoint (e.g., `openai` or a local URL)
    - `--api-key` — Your API key (if required)
    - `--model` — The model name (e.g., `gpt-3.5-turbo`)

3. **Start writing!**
    - Press `Tab` to accept AI completions.
    - Press `Shift+Up/Down` to cycle suggestions.
    - Use arrow keys to move the cursor.

4. **Reflect on the experience**: How does it feel to write with an AI companion? What new forms of creativity emerge?

---

## Further Reading

- [Remarks on AI from NZ](https://nealstephenson.substack.com/p/remarks-on-ai-from-nz)
- [AI, Heidegger, and Evangelion](https://fakepixels.substack.com/p/ai-heidegger-and-evangelion)
- [Zero Cool and the Oraculator](https://nearfuturelaboratory.com/blog/2025/05/zero-cool-and-the-oraculator/)

---

## License

MIT.  
This is an experiment—fork it, remix it, and help invent the future of human-AI collaboration.