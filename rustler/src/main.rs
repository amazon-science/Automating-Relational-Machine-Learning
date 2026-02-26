use clap::{Parser, Subcommand};

mod common;
pub mod fly;
mod pre;
mod encode_unique;
mod gpu_ipca;
mod shallow_pca;
mod text_encoder;
mod text_index;
mod time_cutoffs;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Pre(pre::Cli),
    Fly(fly::Cli),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Pre(cli) => {
            pre::main(cli);
        }
        Command::Fly(cli) => {
            fly::main(cli);
        }
    }
}
