import { ChatOpenAI } from '@langchain/openai'
import * as readline from 'node:readline'
import { ContentBlock, trimMessages, SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage } from '@langchain/core/messages'
import { Calculator } from '@langchain/community/tools/calculator'
import { createAgent } from 'langchain'
import Handlebars from 'handlebars'
import yargs from 'yargs'
import { hideBin } from 'yargs/helpers'
import fs from 'node:fs/promises'
import z from 'zod'
import { BaseCallbackHandler } from '@langchain/core/callbacks/base'

/**
 * Custom callback handler that streams tokens to stdout as they arrive
 */
class StreamingCallbackHandler extends BaseCallbackHandler {
  name = 'StreamingCallbackHandler'

  handleLLMNewToken(token: string): void {
    process.stdout.write(token)
  }
}

// Create streaming callback handler for this invocation
const streamingHandler = new StreamingCallbackHandler()

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0.4,
  apiKey: process.env.CLMI_OPENAI_API_KEY,
  streaming: true,
  // useResponsesApi: true,
})

const calculator = new Calculator()

type BotDefinition = {
  systemMessage: string,
  initialPromptTemplate: string,
}

type CompiledBot = {
  systemMessage: string
  initialPromptTemplate: Handlebars.TemplateDelegate
}

const botDefinitionSchema = z.object({
  systemMessage: z.string(),
  initialPromptTemplate: z.string(),
})

let bot: BotDefinition = {
  systemMessage: 'You are a helpful AI assistant. Responses will be displayed on the command line in plain text so please use minimal formatting.',
  initialPromptTemplate: '{{input}}'
}

let agent = createAgent({
  model: llm,
  tools: [calculator],
  systemPrompt: bot.systemMessage,
})

/*
 * Process command line arguments:
 *
 *  -f, --file <file>  Load the bot definition from a JSON file
 */

const args =
  await yargs(hideBin(process.argv))
    .option('file', { alias: 'f', type: 'string', description: 'Load the bot definition from a JSON file' })
    .parse()

async function loadBotDefinition(file: string): Promise<BotDefinition> {
  return botDefinitionSchema.parse(JSON.parse(await fs.readFile(file, 'utf-8')))
}

if (args.file) {
  try {
    bot = await loadBotDefinition(args.file)
  } catch (e) {
    console.error(`Error loading bot definition from file: ${e instanceof Error ? e.message : e}`)
    process.exit(1)
  }
}

const abortController = new AbortController()
const term = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  tabSize: 2,
  signal: abortController.signal
})

function contentToString(content: string | ContentBlock[]): string {
  if (typeof content === 'string') {
    return content
  }

  const result: any[] = [];

  for (const c of content) {
    switch (c.type) {
      case 'image_url':
      case 'text':
        result.push((<any>c)[c.type])
        break
      default:
        console.warn(`unsupported content type: ${c.type}`)
        break
    }
  }

  return result.length > 1 ?
    JSON.stringify(result, undefined, '  ') :
    (typeof result[0] === 'string' ? result[0] : JSON.stringify(result[0], undefined, '  '))
}

let compiledBot: CompiledBot = {
  systemMessage: bot.systemMessage,
  initialPromptTemplate: Handlebars.compile(bot.initialPromptTemplate)
}

function estimateTokens(content: string | ContentBlock[]) {
  content = contentToString(content)

  return content.split(/\s+/).length
}

/*
 * Configure LangChain history trimmer
 */
const trimmer = trimMessages({
  maxTokens: 1000,
  strategy: 'last',
  allowPartial: true,
  endOn: 'human',
  tokenCounter: (msgs) => {
    const count = msgs.reduce(
      (tokens, msg) => {
        if ('tokenUsage' in msg.response_metadata && 'totalTokens' in (msg.response_metadata['tokenUsage'] as any)) {
          const totalTokens = Number((msg.response_metadata['tokenUsage'] as any)['totalTokens'])
          if (!Number.isNaN(totalTokens)) {
            return totalTokens > tokens ? totalTokens : tokens
          } else {
            console.warn('Invalid token count in response metadata', (msg.response_metadata['tokenUsage'] as any)['totalTokens'])
          }
        } else if (msg instanceof AIMessage) {
          console.warn('Missing token count in AIMessage response metadata', msg.response_metadata)
        }
        return tokens + estimateTokens(msg.content)
      },
      0)
    console.log(`${msgs.length} messages contained ${count} tokens.`)
    return count
  }
})

let conversation: BaseMessage[] = []

/*
 * Line-by-line conversation with the bot
 *
 *   .load - load a new bot definition
 *   .reset - reset the chat history
 *   .exit - exit the app
 */
term.on('line', async (input) => {
  try {
    if (input == null || input.trim() === '.exit') {
      term.close()
      return
    } else if (input.trim() === '.reset') {
      conversation = []
      term.prompt()
      return
    } else if (input.trim().split(' ')[0] === '.load') {
      try {
        bot = (await loadBotDefinition(input.trim().split(' ')[1]))
        compiledBot = {
          systemMessage: bot.systemMessage,
          initialPromptTemplate: Handlebars.compile(bot.initialPromptTemplate)
        }
        agent = createAgent({
          model: llm,
          tools: [calculator],
          systemPrompt: bot.systemMessage,
        })
        conversation = []
        console.info(`Loaded bot definition from file: ${input.trim().split(' ')[1]}`)
      } catch (e) {
        console.error(`Error loading bot definition from file: ${e instanceof Error ? e.message : e}`)
      }
      term.prompt()
      return
    }

    const messages: BaseMessage[] = []
    if (conversation.length == 0) {
      // First user input, use the initial prompt template
      conversation.push(new HumanMessage(compiledBot.initialPromptTemplate({ input })))
      messages.push(conversation[0])
    } else {
      // Subsequent user inputs, use the input as-is, and trim the conversation history to limit token usage
      conversation.push(new HumanMessage(input))
      conversation = await trimmer.invoke(conversation)

      // Show debugging output of the trimmed conversation history
      console.log('[Trimmed Conversation History]\n' + conversation.map(msg => {
        if (msg instanceof HumanMessage) {
          return `    >>> Human: ${contentToString(msg.content)}`
        } else if (msg instanceof AIMessage) {
          if (msg.tool_calls) {
            return msg.tool_calls.map(tc => `       ==> Tool Call: ${tc.name} (${JSON.stringify(tc.args)})`).join('\n')
          } else {
            return `    <<< Bot: ${contentToString(msg.content)}`
          }
        } else if (msg instanceof ToolMessage) {
          return `       <== Tool Result: ${msg.name ?? msg.tool_call_id} (${contentToString(msg.content)})`
        } else {
          return `    ??? ${msg.constructor.name}, ${contentToString(msg.content)}`
        }
      }).join('\n') + '\n[*** End of History ***]')
      messages.push(...conversation)
    }

    // Invoke the agent to generate new output with streaming
    const res = await agent.invoke(
      { messages },
      { callbacks: [streamingHandler] }
    )

    // Write a newline after streaming completes
    process.stdout.write('\n\n')

    // The agent returns all messages including input and new ones
    // Find messages that are new (after our input messages)
    const inputMessageCount = messages.length
    const newMessages = res.messages.slice(inputMessageCount)
    
    // Add all new messages (tool calls, tool results, final response) to conversation
    conversation.push(...newMessages)

    // Prompt the user for the next input
    term.prompt()
  } catch (e) {
    console.error('Unexpected error processing user input', e)
    process.exit(1)
  }
})

term.prompt()
