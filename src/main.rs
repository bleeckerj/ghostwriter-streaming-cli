#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused)]
use std::io;
use std::time::{Duration, Instant};
use async_openai::types::*;
use async_openai::Client;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers, KeyEvent},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures_util::{future::join_all, StreamExt};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Position},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use tokio::sync::mpsc;
use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Model API endpoint to use: "openai" or a custom URL for LM Studio
    #[arg(short, long, default_value = "openai")]
    endpoint: String,
    
    /// Model name to use for OpenAI API
    #[arg(short, long, default_value = "gpt-4.1-mini-2025-04-14")]
    model: String,
    
    /// OpenAI API key (only needed for OpenAI endpoint)
    #[arg(short, long)]
    api_key: Option<String>,
    
    /// LM Studio model name (only used when endpoint is not "openai")
    #[arg(long, default_value = "llama-3.2-3b-instruct")]
    lm_model: String,
}

struct App {
    input: String,
    completions: Vec<String>,
    original_completions: Vec<String>,
    current_completion_index: usize,
    last_keypress: Instant,
    completion_in_progress: bool,
    cursor_position: usize, // Byte offset in input
    cursor_line: usize,     // Line number (0-based)
    cursor_col: usize,      // Column in line (0-based, in chars)
    scroll_line_offset: usize, // First visible line in viewport
    debug_message: String,
    waiting_for_user_input: bool,
    editor_width: usize,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            completions: Vec::new(),
            original_completions: Vec::new(),
            current_completion_index: 0,
            last_keypress: Instant::now(),
            completion_in_progress: false,
            cursor_position: 0,
            cursor_line: 0,
            cursor_col: 0,
            scroll_line_offset: 0,
            debug_message: String::new(),
            waiting_for_user_input: false,
            editor_width: 0,
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
    
    // Add methods to handle cursor movement
    fn move_cursor_left(&mut self) {
        // Handle multi-byte characters when moving left
        if self.cursor_position > 0 {
            let previous_char_boundary = self.input[..self.cursor_position]
            .char_indices()
            .last()
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
            self.cursor_position = previous_char_boundary;
        }
    }
    
    fn move_cursor_right(&mut self) {
        // Get the next character boundary
        if self.cursor_position < self.input.len() {
            let next_char_boundary = self.input[self.cursor_position..]
            .chars()
            .next()
            .map(|c| self.cursor_position + c.len_utf8())
            .unwrap_or(self.cursor_position);
            
            self.cursor_position = next_char_boundary;
        }
    }
    
    // Add this new method to App
    fn should_load_more(&self) -> bool {
        // Trigger "load more" when we're at the last item or close to it
        !self.completions.is_empty() && 
        self.current_completion_index >= self.completions.len().saturating_sub(1)
    }
    
    fn update_cursor_line_col(&mut self) {
        let mut line = 0;
        let mut col = 0;
        let mut count = 0;
        for l in self.input.split('\n') {
            let line_len = l.chars().count() + 1; // +1 for '\n'
            if self.cursor_position < count + line_len {
                line = line;
                col = self.cursor_position - count;
                break;
            }
            count += line_len;
            line += 1;
        }
        self.cursor_line = line;
        self.cursor_col = col;
    }
    
    fn set_cursor_from_line_col(&mut self) {
        let mut pos = 0;
        let mut line = 0;
        for l in self.input.split('\n') {
            if line == self.cursor_line {
                let col = self.cursor_col.min(l.chars().count());
                pos += l.chars().take(col).map(|c| c.len_utf8()).sum::<usize>();
                break;
            }
            pos += l.len() + 1; // +1 for '\n'
            line += 1;
        }
        self.cursor_position = pos;
    }
    
    /// Returns a Vec of (start_byte, end_byte) for each visual line in the current logical line.
    fn get_visual_lines(&self, line: &str, width: usize) -> Vec<(usize, usize)> {
        let mut visual_lines = Vec::new();
        let mut start = 0;
        let mut current_width = 0;
        let mut last_break = 0;
        for (idx, c) in line.char_indices() {
            current_width += 1;
            if current_width > width {
                visual_lines.push((start, idx));
                start = idx;
                current_width = 1;
            }
            last_break = idx + c.len_utf8();
        }
        // Push the last segment
        if start < line.len() {
            visual_lines.push((start, line.len()));
        }
        visual_lines
    }
    
    /// Returns (visual_line_idx, col_in_visual) for the current cursor_col in the logical line.
    fn get_visual_cursor(&self, line: &str, width: usize, cursor_col: usize) -> (usize, usize) {
        let mut visual_idx = 0;
        let mut col_in_visual = cursor_col;
        let mut count = 0;
        let mut chars = line.chars();
        while count < cursor_col {
            let mut visual_width = 0;
            let mut visual_count = 0;
            while visual_width < width && count < cursor_col {
                if let Some(_) = chars.next() {
                    visual_width += 1;
                    visual_count += 1;
                    count += 1;
                } else {
                    break;
                }
            }
            if count < cursor_col {
                visual_idx += 1;
                col_in_visual -= visual_count;
            }
        }
        (visual_idx, col_in_visual)
    }
    
    /// Move the cursor up by one visual line (wrap row).
    pub fn move_cursor_up_visual(&mut self, width: usize) {
        let line = self.input.split('\n').nth(self.cursor_line).unwrap_or("");
        let visual_lines = self.get_visual_lines(line, width);
        let (visual_idx, col_in_visual) = self.get_visual_cursor(line, width, self.cursor_col);
        if visual_idx > 0 {
            // Move to previous visual line, clamp col
            let (start, end) = visual_lines[visual_idx - 1];
            let prev_len = line[start..end].chars().count();
            let new_col = start + col_in_visual.min(prev_len);
            self.cursor_col = line[..new_col].chars().count();
            self.set_cursor_from_line_col();
        } else if self.cursor_line > 0 {
            // Move to end of previous logical line
            self.cursor_line -= 1;
            let prev_line = self.input.split('\n').nth(self.cursor_line).unwrap_or("");
            let prev_visual_lines = self.get_visual_lines(prev_line, width);
            let (start, end) = *prev_visual_lines.last().unwrap();
            let prev_len = prev_line[start..end].chars().count();
            self.cursor_col = prev_line.len().min(prev_line[..end].chars().count());
            self.set_cursor_from_line_col();
        }
    }
    
    /// Move the cursor down by one visual line (wrap row).
    pub fn move_cursor_down_visual(&mut self, width: usize) {
        let line = self.input.split('\n').nth(self.cursor_line).unwrap_or("");
        let visual_lines = self.get_visual_lines(line, width);
        let (visual_idx, col_in_visual) = self.get_visual_cursor(line, width, self.cursor_col);
        if visual_idx + 1 < visual_lines.len() {
            // Move to next visual line, clamp col
            let (start, end) = visual_lines[visual_idx + 1];
            let next_len = line[start..end].chars().count();
            let new_col = start + col_in_visual.min(next_len);
            self.cursor_col = line[..new_col].chars().count();
            self.set_cursor_from_line_col();
        } else if self.cursor_line + 1 < self.input.lines().count() {
            // Move to start of next logical line
            self.cursor_line += 1;
            self.cursor_col = 0;
            self.set_cursor_from_line_col();
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    
    // Parse command line arguments
    let args = Args::parse();
    
    let api_key = match args.endpoint.as_str() {
        "openai" => {
            // For OpenAI, we need an API key
            args.api_key.clone().or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .ok_or_else(|| anyhow::anyhow!("OpenAI API key not found. Please provide it with --api-key or set the OPENAI_API_KEY environment variable."))?
        },
        _ => {
            // For other endpoints like LM Studio, the key might not be required
            args.api_key.clone().unwrap_or_else(|| "no-api-key-required".to_string())
        }
    };
    
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
                Constraint::Min(1),      // Main editor area
                Constraint::Length(1),   // Yellow separator
                Constraint::Length(8),   // Debug area (2 lines tall)
                ])
                .split(size);
                
                app.editor_width = chunks[0].width as usize - 2; // minus borders
                
                let lines: Vec<&str> = app.input.split('\n').collect();
                let max_visible_lines = chunks[0].height as usize - 2; // minus borders
                
                // Adjust scroll offset
                if app.cursor_line < app.scroll_line_offset {
                    app.scroll_line_offset = app.cursor_line;
                } else if app.cursor_line >= app.scroll_line_offset + max_visible_lines {
                    app.scroll_line_offset = app.cursor_line - max_visible_lines + 1;
                }
                
                // Get visible lines
                let visible_lines = &lines[app.scroll_line_offset..app.scroll_line_offset + max_visible_lines.min(lines.len() - app.scroll_line_offset)];
                
                // Build styled lines
                let mut styled_lines = Vec::new();
                for (i, line) in visible_lines.iter().enumerate() {
                    let line_idx = app.scroll_line_offset + i;
                    if line_idx == app.cursor_line {
                        // Highlight cursor position
                        let col = app.cursor_col.min(line.chars().count());
                        let (before, at, after) = {
                            let mut before = String::new();
                            let mut at = String::new();
                            let mut after = String::new();
                            let mut chars = line.chars();
                            for _ in 0..col {
                                if let Some(c) = chars.next() {
                                    before.push(c);
                                }
                            }
                            if let Some(c) = chars.next() {
                                at.push(c);
                            } else {
                                at.push(' ');
                            }
                            after = chars.collect();
                            (before, at, after)
                        };
                        styled_lines.push(Line::from(vec![
                            Span::raw(before),
                            Span::styled(at, Style::default().bg(Color::White).fg(Color::Black)),
                            Span::raw(after),
                            Span::styled(
                                app.current_completion(),
                                Style::default().fg(Color::Blue).add_modifier(Modifier::ITALIC),
                            ),
                            ]));
                        } else {
                            styled_lines.push(Line::from(vec![Span::raw(*line)]));
                        }
                    }
                    
                    let paragraph = Paragraph::new(styled_lines)
                    .block(Block::default().borders(Borders::ALL).title("Ghostwriter UX Exploration CLI"))
                    .wrap(Wrap { trim: false }); // soft wrapping
                    
                    f.render_widget(paragraph, chunks[0]);
                    
                    // Yellow separator line
                    let separator = Line::from(vec![Span::styled(
                        "─".repeat(chunks[1].width as usize),
                        Style::default().fg(Color::Yellow),
                    )]);
                    f.render_widget(Paragraph::new(separator), chunks[1]);
                    
                    // Debug area
                    let debug = Paragraph::new(app.debug_message.clone())
                    .block(Block::default().borders(Borders::ALL).title("Debug"))
                    .style(Style::default().fg(Color::Yellow));
                    f.render_widget(debug, chunks[2]);
                    
                    // After rendering the main editor area (paragraph)
                    let current_line = lines.get(app.cursor_line).unwrap_or(&"");
                    let editor_width = chunks[0].width as usize - 2; // minus borders
                    
                    // Calculate visual position accounting for wrapping
                    let visual_line = app.cursor_col / editor_width;
                    let visual_col = app.cursor_col % editor_width;
                    
                    // Simple cursor positioning that matches soft-wrapped rendering
                    let base_y = chunks[0].y + 1;
                    let line_offset = if app.cursor_line >= app.scroll_line_offset {
                        app.cursor_line - app.scroll_line_offset
                    } else {
                        0
                    };
                    
                    // Don't try to calculate wrapping - let the terminal handle it naturally
                    let cursor_y = base_y.saturating_add(line_offset as u16);
                    let cursor_x = chunks[0].x + 1 + app.cursor_col as u16;
                    
                    // Clamp to editor bounds
                    let max_x = chunks[0].x + chunks[0].width - 2;
                    let cursor_x = cursor_x.min(max_x);
                    
                    f.set_cursor_position(ratatui::layout::Position { x: cursor_x, y: cursor_y });
                    app.debug_message = format!(
                        "cursor_x: {}, cursor_y: {}, line: {}, col: {}, pos: {}",
                        cursor_x, cursor_y, app.cursor_line, app.cursor_col, app.cursor_position
                    );
                })?;
                
                if event::poll(Duration::from_millis(100))? {
                    if let Event::Key(key) = event::read()? {
                        match key {
                            KeyEvent {
                                code: KeyCode::Up,
                                modifiers: KeyModifiers::SHIFT,
                                ..
                            } => {
                                // Ctrl-Up: previous completion
                                if !app.completions.is_empty() {
                                    if app.current_completion_index == 0 {
                                        app.current_completion_index = app.completions.len() - 1;
                                    } else {
                                        app.current_completion_index -= 1;
                                    }
                                }
                            }
                            KeyEvent {
                                code: KeyCode::Down,
                                modifiers: KeyModifiers::SHIFT,
                                ..
                            } => {
                                // Ctrl-Down: next completion
                                app.next_completion();
                                // If we're at or near the end, trigger load more
                                if app.should_load_more() && !app.completion_in_progress {
                                    let prompt = app.input.clone();
                                    let tx_clone = tx.clone();
                                    app.completion_in_progress = true;
                                    let args_clone = args.clone();
                                    let api_key_clone = api_key.clone();
                                    tokio::spawn(async move {
                                        let effective_model = if args_clone.endpoint == "openai" {
                                            args_clone.model
                                        } else {
                                            args_clone.lm_model
                                        };
                                        if let Ok(completions) = stream_multiple_openai_completions(
                                            &args_clone.endpoint, &effective_model, &api_key_clone, &prompt, 3
                                        ).await {
                                            if !completions.is_empty() {
                                                let _ = tx_clone.send(completions).await;
                                            }
                                        }
                                    });
                                }
                            }
                            
                            KeyEvent { code: KeyCode::Char(c), modifiers, .. }
                            if !modifiers.contains(KeyModifiers::CONTROL) && !modifiers.contains(KeyModifiers::ALT) =>
                            {
                                // Accept all normal typing, including Shift and CapsLock
                                app.input.insert(app.cursor_position, c);
                                app.cursor_position += 1;
                                app.update_cursor_line_col();
                                
                                // Reset the flag here:
                                if app.waiting_for_user_input {
                                    app.waiting_for_user_input = false;
                                }
                                
                                if !app.completions.is_empty() && app.current_completion().starts_with(c) {
                                    let current_idx = app.current_completion_index;
                                    let mut completion = app.completions[current_idx].clone();
                                    completion.remove(0);
                                    app.completions[current_idx] = completion;
                                } else {
                                    app.completions.clear();
                                    app.original_completions.clear();
                                }
                                app.last_keypress = Instant::now();
                            }
                            KeyEvent { code: KeyCode::Backspace, modifiers: KeyModifiers::NONE, .. } => {
                                if app.cursor_position > 0 {
                                    app.input.remove(app.cursor_position - 1);
                                    app.cursor_position -= 1;
                                    app.update_cursor_line_col();
                                }
                                if !app.original_completions.is_empty() {
                                    let current_idx = app.current_completion_index;
                                    if current_idx < app.original_completions.len() {
                                        let original = &app.original_completions[current_idx];
                                        if original.starts_with(&app.input) {
                                            app.completions[current_idx] = original[app.input.len()..].to_string();
                                        }
                                    }
                                } else {
                                    app.completions.clear();
                                }
                                app.last_keypress = Instant::now();
                            }
                            KeyEvent { code: KeyCode::Delete, modifiers: KeyModifiers::NONE, .. } => {
                                if app.cursor_position < app.input.len() {
                                    // Delete character at cursor position
                                    app.input.remove(app.cursor_position);
                                }
                                // Reset completions as content changed
                                app.completions.clear();
                                app.original_completions.clear();
                                app.last_keypress = Instant::now();
                            }
                            KeyEvent { code: KeyCode::Tab, modifiers: KeyModifiers::NONE, .. } => {
                                if !app.current_completion().is_empty() {
                                    let current_completion = app.current_completion().to_string();
                                    if !current_completion.is_empty() {
                                        app.input.insert_str(app.cursor_position, &current_completion);
                                        app.cursor_position += current_completion.len();
                                        app.update_cursor_line_col();
                                    }
                                    app.completions.clear();
                                    app.original_completions.clear();
                                    app.current_completion_index = 0;
                                    app.waiting_for_user_input = true; // <-- Add this
                                }
                            }
                            KeyEvent { code: KeyCode::Home, modifiers: KeyModifiers::NONE, .. } => {
                                app.cursor_position = 0;
                            }
                            KeyEvent { code: KeyCode::End, modifiers: KeyModifiers::NONE, .. } => {
                                if app.cursor_position == app.input.len() && !app.current_completion().is_empty() {
                                    let current_completion = app.current_completion().to_string();
                                    app.input.push_str(&current_completion);
                                    app.cursor_position += current_completion.len();
                                    app.completions.clear();
                                    app.original_completions.clear();
                                    app.current_completion_index = 0;
                                } else {
                                    app.cursor_position = app.input.len();
                                }
                            }
                            KeyEvent { code: KeyCode::Esc, .. } => {
                                app.debug_message.push_str(stringify!("\nExiting..."));
                                break
                            }
                            KeyEvent { code: KeyCode::Up, modifiers: KeyModifiers::NONE, .. } => {
                                app.move_cursor_up_visual(app.editor_width);
                            }
                            KeyEvent { code: KeyCode::Down, modifiers: KeyModifiers::NONE, .. } => {
                                app.move_cursor_down_visual(app.editor_width);
                            }
                            KeyEvent { code: KeyCode::Left, modifiers: KeyModifiers::NONE, .. } => {
                                if app.cursor_col > 0 {
                                    app.cursor_col -= 1;
                                    app.set_cursor_from_line_col();
                                } else if app.cursor_line > 0 {
                                    app.cursor_line -= 1;
                                    app.cursor_col = app.input.lines().nth(app.cursor_line).unwrap_or("").chars().count();
                                    app.set_cursor_from_line_col();
                                }
                            }
                            KeyEvent { code: KeyCode::Right, modifiers: KeyModifiers::NONE, .. } => {
                                let line_len = app.input.lines().nth(app.cursor_line).unwrap_or("").chars().count();
                                if app.cursor_col < line_len {
                                    app.cursor_col += 1;
                                    app.set_cursor_from_line_col();
                                } else if app.cursor_line + 1 < app.input.lines().count() {
                                    app.cursor_line += 1;
                                    app.cursor_col = 0;
                                    app.set_cursor_from_line_col();
                                }
                            }
                            KeyEvent { code: KeyCode::Enter, modifiers: KeyModifiers::NONE, .. } => {
                                app.input.insert(app.cursor_position, '\n');
                                app.cursor_position += 1;
                                app.update_cursor_line_col();
                            }
                            _ => {}
                        }
                    }
                }
                
                if app.last_keypress.elapsed() > Duration::from_millis(600)
                && !app.input.is_empty()
                && app.completions.is_empty()
                && !app.completion_in_progress
                && !app.waiting_for_user_input // waiting for the user to type something
                
                {
                    let prompt = app.input.clone();
                    let tx_clone = tx.clone();
                    app.completion_in_progress = true;
                    let endpoint = args.endpoint.clone();
                    let model = args.model.clone();
                    let lm_model = args.lm_model.clone(); // Clone the LM model separately
                    let api_key_clone = api_key.clone(); // Clone the API key
                    tokio::spawn(async move {
                        // Choose appropriate model based on endpoint
                        let effective_model = if endpoint == "openai" {
                            model
                        } else {
                            lm_model  // Use LM Studio model here
                        };
                        
                        if let Ok(completions) = stream_multiple_openai_completions(&endpoint, &effective_model, &api_key_clone, &prompt, 3).await {
                            if !completions.is_empty() {
                                let _ = tx_clone.send(completions).await;
                            }
                        }
                    });
                }
                
                // Check if the completion task has finished:
                if let Ok(new_completions) = rx.try_recv() {
                    app.debug_message = format!("Received {} completions", new_completions.len());
                    app.debug_message.push_str("\nCompletions: ");
                    app.debug_message.push_str(&new_completions.join(", "));
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
        
        async fn stream_openai_completion(endpoint: &str, model: &str, api_key: &str, prompt: &str) -> anyhow::Result<String> {
            // This function remains mostly the same, but log the model being used
            if endpoint != "openai" {
                //println!("Using LM Studio model: {}", model);
            }
            
            let client = if endpoint == "openai" {
                // OpenAI configuration remains unchanged
                let config = async_openai::config::OpenAIConfig::new()
                .with_api_key(api_key);
                Client::with_config(config)
            } else {
                // For LM Studio, add more verbose logging
                //println!("Connecting to LM Studio at: {}", endpoint);
                let config = async_openai::config::OpenAIConfig::new()
                .with_api_base(endpoint)
                .with_api_key(api_key);
                
                Client::with_config(config)
            };
            
            let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a helpful and creative assistant that completes partial thoughts. Only continue the user’s input as plain prose in the same tone and register. Do not prepend or append ellipses, quotation marks, or any special characters. Your response is only the continuation — e.g.
                    Input: “The sun was just rising over the hills”
                    Output: “and the dew still clung to the grass, sparkling like glass.” 
                    Never begin with “...” or any symbols.
                    Do not wrap the output with quotation marks or any other characters. 
                    Never begin with a quotation mark. You are completing text, not provide a quotation.
                    Do not use any special formatting like bullet points, lists, or numbered items.
                    Do not use any special characters like ellipses, quotation marks, or dashes.
                    Do not repeat the user’s input. Do not add any additional commentary or explanation.
                    Your output should have an Automated Readability Index of 1 or lower. This is not a chat. Do not answer questions.")
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
                
                // Add more verbose debugging for LM Studio
                if endpoint != "openai" {
                    //println!("Stream initialized, waiting for tokens...");
                }
                
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(response) => {
                            if let Some(fragment) = response.choices.first().and_then(|c| c.delta.content.clone()) {
                                output.push_str(&fragment);
                                
                                // For debugging LM Studio responses
                                if endpoint != "openai" {
                                    //println!("Received fragment: {}", fragment);
                                }
                            } else if endpoint != "openai" {
                                // Debug when we got a response but no content
                                //println!("Received response but no content in delta: {:?}", response);
                            }
                        },
                        Err(e) => {
                            // Print error but continue
                            eprintln!("Error in stream: {}", e);
                        }
                    }
                }
                
                if output.is_empty() && endpoint != "openai" {
                    // Special handling for empty responses from LM Studio
                    //println!("No output received from LM Studio");
                    
                    // Try non-streaming as fallback (some LM Studio setups might not support streaming properly)
                    //println!("Attempting non-streaming fallback...");
                    
                    let non_stream_request = CreateChatCompletionRequestArgs::default()
                    .model(model)
                    .messages(vec![
                        ChatCompletionRequestMessage::System(
                            ChatCompletionRequestSystemMessageArgs::default()
                            .content("You are a helpful and creative assistant that completes partial thoughts. Complete the user's input with 3-5 sentences.")
                            .build()?,
                        ),
                        ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessageArgs::default()
                            .content(prompt.to_string())
                            .build()?,
                        ),
                        ])
                        .build()?;
                        
                        match client.chat().create(non_stream_request).await {
                            Ok(response) => {
                                if let Some(choice) = response.choices.first() {
                                    if let Some(content) = &choice.message.content {
                                        //println!("Non-streaming response: {}", content);
                                        output = content.clone();
                                    }
                                }
                            },
                            Err(e) => eprintln!("Non-streaming fallback failed: {}", e)
                        }
                    }
                    
                    Ok(output)
                }
                
                
                async fn stream_multiple_openai_completions(endpoint: &str, model: &str, api_key: &str, prompt: &str, num_completions: usize) -> anyhow::Result<Vec<String>> {
                    let futures = (0..num_completions).map(|_| {
                        let prompt = prompt.to_string();
                        let endpoint = endpoint.to_string();
                        let model = model.to_string();
                        let api_key = api_key.to_string();
                        
                        tokio::spawn(async move {
                            stream_openai_completion(&endpoint, &model, &api_key, &prompt).await
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