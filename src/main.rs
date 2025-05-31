#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused)]
use std::io;
use std::time::{Duration, Instant};
use async_openai::types::*;
use async_openai::Client;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::{future::join_all, StreamExt};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use tokio::sync::mpsc;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model API endpoint to use: "openai" or a custom URL for LM Studio
    #[arg(short, long, default_value = "openai")]
    endpoint: String,

    /// Model name to use
    #[arg(short, long, default_value = "gpt-4o")]
    model: String,
}

struct App {
    input: String,
    completions: Vec<String>,
    original_completions: Vec<String>,  // <-- Add this to store original completions
    current_completion_index: usize,
    last_keypress: Instant,
    completion_in_progress: bool,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            completions: Vec::new(),
            original_completions: Vec::new(),  // <-- Initialize
            current_completion_index: 0,
            last_keypress: Instant::now(),
            completion_in_progress: false,
        }
    }

    fn current_completion(&self) -> &str {
        self.completions.get(self.current_completion_index).map(|s| s.as_str()).unwrap_or("")
    }

    fn next_completion(&mut self) {
        if !self.completions.is_empty() {
            self.current_completion_index = (self.current_completion_index + 1) % self.completions.len();
        }
    }

    // Add this new method to App
    fn should_load_more(&self) -> bool {
        // Trigger "load more" when we're at the last item or close to it
        !self.completions.is_empty() && 
        self.current_completion_index >= self.completions.len().saturating_sub(1)
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
    let (tx, mut rx) = mpsc::channel::<Vec<String>>(1);

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
                Span::styled(app.current_completion(), Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC)),
            ];
            let paragraph = Paragraph::new(Line::from(spans))
                .block(Block::default().borders(Borders::ALL).title("Ghostwriter UX Exploration CLI"))
                .wrap(Wrap { trim: true }); // Enable wrapping here

            f.render_widget(paragraph, chunks[0]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char(c) => {
                        app.input.push(c);
                        
                        // If we have completions and the typed character matches the next character
                        // in the current completion, remove it from the completion
                        if !app.completions.is_empty() && app.current_completion().starts_with(c) {
                            let current_idx = app.current_completion_index;
                            let mut completion = app.completions[current_idx].clone();
                            completion.remove(0); // Remove the matched character
                            app.completions[current_idx] = completion;
                        } else {
                            // If mismatch or no completions, clear them
                            app.completions.clear();
                            app.original_completions.clear();
                        }
                        
                        app.last_keypress = Instant::now();
                    }
                    KeyCode::Backspace => {
                        if !app.input.is_empty() {
                            app.input.pop();
                            
                            // If we have original completions, check if we can restore
                            if !app.original_completions.is_empty() {
                                let current_idx = app.current_completion_index;
                                if current_idx < app.original_completions.len() {
                                    let original = &app.original_completions[current_idx];
                                    
                                    // If input is a prefix of the original completion source
                                    if original.starts_with(&app.input) {
                                        // Restore the completion for this position
                                        app.completions[current_idx] = original[app.input.len()..].to_string();
                                    }
                                }
                            } else {
                                app.completions.clear();
                            }
                        }
                        
                        app.last_keypress = Instant::now();
                    }
                    KeyCode::Tab => {
                        if !app.current_completion().is_empty() {
                            let current_completion = app.current_completion().to_string();
                            if !currentCompletion.is_empty() {
                                app.input.push_str(&currentCompletion);
                            }
                            app.completions.clear();
                            app.original_completions.clear();
                            app.current_completion_index = 0;
                        }
                    }
                    KeyCode::Down => {
                        app.next_completion(); // cycle through completions
                        
                        // If we're at or near the end of available completions, request more
                        if app.should_load_more() && !app.completion_in_progress {
                            let prompt = app.input.clone();
                            let tx_clone = tx.clone();
                            app.completion_in_progress = true;
                            
                            tokio::spawn(async move {
                                if let Ok(new_completions) = stream_multiple_openai_completions(&prompt, 3).await {
                                    if !new_completions.is_empty() {
                                        let _ = tx_clone.send(new_completions).await;
                                    }
                                }
                            });
                        }
                    }
                    KeyCode::Up => {
                        if !app.completions.is_empty() {
                            // Go to previous completion or wrap to the end
                            app.current_completion_index = if app.current_completion_index > 0 {
                                app.current_completion_index - 1
                            } else {
                                app.completions.len() - 1
                            };
                        }
                    }
                    KeyCode::Esc => break,
                    _ => {}
                }
            }
        }

        if app.last_keypress.elapsed() > Duration::from_millis(600)
            && !app.input.is_empty()
            && app.completions.is_empty()
            && !app.completion_in_progress
        {
            let prompt = app.input.clone();
            let tx_clone = tx.clone();
            app.completion_in_progress = true;

            tokio::spawn(async move {
                if let Ok(completions) = stream_multiple_openai_completions(&prompt, 3).await {
                    if !completions.is_empty() {
                        let _ = tx_clone.send(completions).await;
                    }
                }
            });
        }

        // Check if the completion task has finished:
        if let Ok(new_completions) = rx.try_recv() {
            if app.completions.is_empty() {
                // First set of completions
                app.completions = new_completions.clone();
                app.original_completions = new_completions;
                app.current_completion_index = 0;
            } else {
                // Add new completions to existing ones
                app.completions.extend(new_completions.clone());
                app.original_completions.extend(new_completions);
            }
            app.completion_in_progress = false;
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

async fn stream_openai_completion(endpoint: &str, model: &str, prompt: &str) -> anyhow::Result<String> {
    let client = if endpoint == "openai" {
        // Use default OpenAI configuration
        Client::<async_openai::config::OpenAIConfig>::new()
    } else {
        // Use custom endpoint (LM Studio)
        let config = async_openai::config::Config::new()
            .with_api_base(endpoint)
            .with_api_key("no-api-key-required"); // LM Studio typically doesn't require an API key
        
        Client::with_config(config)
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(vec![
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a helpful and creative assistant that completes partial thoughts. You should only respond with the completion of the user's input with no more than 2-5 sentences. The response should have a Automated Readability Index of no more than 2.")
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

    let mut stream = client.chat().create_stream(request).await?;

    let mut output = String::new();
    while let Some(token) = stream.next().await {
        if let Some(fragment) = token?.choices.first().and_then(|c| c.delta.content.clone()) {
            output.push_str(&fragment);
        }
    }

    Ok(output)
}


async fn stream_multiple_openai_completions(prompt: &str, num_completions: usize) -> anyhow::Result<Vec<String>> {
    //let client = Client::new();

    let futures = (0..num_completions).map(|_| {
        let prompt = prompt.to_string();
        //let client = client.clone();
        tokio::spawn(async move {
            stream_openai_completion(&prompt).await
        })
    });

    let results = join_all(futures).await;

    // Collect successful completions
    let completions: Vec<String> = results.into_iter()
        .filter_map(|res| res.ok())
        .filter_map(|res| res.ok())
        .collect();

    Ok(completions)
}