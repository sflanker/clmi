# CLMI

## Overview

CLMI (CLI Language Model Interface) is a command-line tool that allows users to interact with a large language model (LLM) using a conversational interface.
The tool leverages the LangChain library to manage conversation history and the OpenAI API to generate responses.
Users can load custom bot definitions, reset the chat history, and exit the application using specific commands.

## Build Instructions

### Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/sflanker/clmi.git
   cd clmi
   ```

2. Install dependencies:
   ```sh
   npm install
   ```

### Build

To build the project, run:
```sh
npm run build
```

## Usage

### Environment Setup

In order for `clmi` to connect to OpenAI it is necessary to set the `CLMI_OPENAI_API_KEY` to your OpenAI API key.
You can set this in your shell profile, or create a `.env` file by copying `.env.example` and set your key there.
With the `.env` file in place you can either us a tool like [direnv](https://direnv.net/), or you can use `npm run with-env -- npm start` when launching `clmi`.

### Running the Program

To start the CLMI tool, run:
```sh
npm start
```

### Interacting with the Program

Once the program is running, you can interact with it using the command line. The following commands are available:

- **Load a new bot definition**: Type `.load <path-to-bot-definition.json>` and press Enter.
- **Reset the chat history**: Type `.reset` and press Enter.
- **Exit the application**: Type `.exit` and press Enter.
