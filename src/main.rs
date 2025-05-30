use std::io;
use std::time::{Duration, Instant};
use anyhow::Result;
use async_openai::types::*;
use async_openai::Client;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::StreamExt;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use tokio::sync::mpsc;

struct App {
    input: String,
    current_completion: String,
    original_completion: String, // <-- Add this
    last_keypress: Instant,
    completion_in_progress: bool,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            current_completion: String::new(),
            original_completion: String::new(), // <-- Initialize here
            last_keypress: Instant::now(),
            completion_in_progress: false,
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let (tx, mut rx) = mpsc::channel::<String>(1);

    loop {
        terminal.draw(|f| {
            let size = f.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(2)
                .constraints([
                    Constraint::Min(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                ])
                .split(size);

            let spans = vec![
                Span::raw(&app.input),
                Span::styled(&app.current_completion, Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC)),
            ];
            let paragraph = Paragraph::new(Line::from(spans))
                .block(Block::default().borders(Borders::ALL).title("Streaming CLI"))
                .wrap(Wrap { trim: true }); // Enable wrapping here

            f.render_widget(paragraph, chunks[0]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char(c) => {
                        app.input.push(c);
                        if app.current_completion.starts_with(c) {
                            app.current_completion.remove(0);
                        } else {
                            app.current_completion.clear();
                        }
                        app.last_keypress = Instant::now();
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                        // Restore completion if input matches original completion again
                        if app.original_completion.starts_with(&app.input) {
                            app.current_completion = app.original_completion[app.input.len()..].to_string();
                        } else {
                            app.current_completion.clear();
                        }
                        app.last_keypress = Instant::now();
                    }
                    KeyCode::Tab => {
                        if !app.current_completion.is_empty() {
                            app.input.push_str(&app.current_completion);
                            app.current_completion.clear();
                        }
                    }
                    KeyCode::Esc => break,
                    _ => {}
                }
            }
        }

        if app.last_keypress.elapsed() > Duration::from_millis(600)
            && !app.input.is_empty()
            && app.current_completion.is_empty()
            && !app.completion_in_progress
        {
            let prompt = app.input.clone();
            let tx_clone = tx.clone();
            app.completion_in_progress = true;

            tokio::spawn(async move {
                if let Ok(completion) = stream_openai_completion(&prompt).await {
                    let _ = tx_clone.send(completion).await;
                }
            });
        }

        // Check if the completion task has finished:
        if let Ok(completion) = rx.try_recv() {
            app.current_completion = completion.clone();
            app.original_completion = completion; // <-- store original completion
            app.completion_in_progress = false;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

async fn stream_openai_completion(prompt: &str) -> anyhow::Result<String> {
    let request = CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages(vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a helpful and creative assistant that completes partial thoughts. You should only respond with the completion of the user's input with no more than 2-5 sentences.")
                    .build()?,
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt.to_string())
                    .build()?,
            ),
        ])
        .stream(true)
        .build()?;

    let client = Client::new();
    let mut stream = client.chat().create_stream(request).await?;

    let mut output = String::new();
    while let Some(token) = stream.next().await {
        if let Some(fragment) = token?.choices.first().and_then(|c| c.delta.content.clone()) {
            output.push_str(&fragment);
        }
    }

    Ok(output)
}
