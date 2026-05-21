## OpenRouter Chat Completions API wrapper.
##
## Drop-in replacement for bitworld/ais/openai that routes requests to
## OpenRouter (https://openrouter.ai/api/v1/chat/completions) instead.
## Exports the same symbols: aiKey, ConversationMessage, TalkResult,
## startTalkToAI, pollTalkToAI, talkToAI.
##
## Set OPENROUTER_API_KEY in the environment before running.
## The model is controlled by OPENROUTER_MODEL (default: claude-haiku).

import
  std/[json, options, os, strutils],
  curly, jsony

const
  DefaultOpenRouterTimeoutSeconds = 30
  DefaultModel = "anthropic/claude-haiku-4-5"

var
  aiKey* = getEnv("OPENROUTER_API_KEY")

let
  aiBaseUrl = "https://openrouter.ai/api/v1/chat/completions"
  aiModel = block:
    let m = getEnv("OPENROUTER_MODEL").strip()
    if m.len > 0: m else: DefaultModel
  curl = newCurly(3)

type
  ConversationMessage* = object
    role*: string
    content*: string

  TalkResult* = object
    done*: bool
    ok*: bool
    tag*: string
    reply*: string
    error*: string

  ChatRequest = ref object
    model: string
    messages: seq[ConversationMessage]
    max_tokens: int

proc timeoutSeconds(): int =
  let v = getEnv("OPENROUTER_TIMEOUT_SECONDS").strip()
  if v.len == 0: return DefaultOpenRouterTimeoutSeconds
  try: max(1, int(parseFloat(v)))
  except ValueError: DefaultOpenRouterTimeoutSeconds

proc requestBody(messages: openArray[ConversationMessage]): string =
  var msgs: seq[ConversationMessage]
  for m in messages: msgs.add(m)
  let req = ChatRequest(model: aiModel, messages: msgs, max_tokens: 200)
  req.toJson()

proc requestHeaders(): HttpHeaders =
  result["Authorization"] = "Bearer " & aiKey
  result["Content-Type"] = "application/json"
  result["HTTP-Referer"] = "https://github.com/Metta-AI/bitworld"
  result["X-Title"] = "lessandro-nirvana Among Them"

proc parseReply(body: string): string =
  let data = parseJson(body)
  let choices = data{"choices"}
  if choices.isNil or choices.len == 0:
    return ""
  let msg = choices[0]{"message"}
  if msg.isNil: return ""
  msg{"content"}.getStr()

proc startTalkToAI*(
  messages: openArray[ConversationMessage],
  tag = ""
): bool =
  ## Starts a non-blocking OpenRouter request.
  if aiKey.len == 0:
    return false
  curl.startRequest(
    "POST",
    aiBaseUrl,
    requestHeaders(),
    requestBody(messages),
    timeoutSeconds(),
    tag
  )
  true

proc pollTalkToAI*(): TalkResult =
  ## Polls for one finished non-blocking request.
  let answer = curl.pollForResponse()
  if answer.isNone:
    return TalkResult(done: false)
  result.done = true
  result.tag = answer.get.response.request.tag
  if answer.get.error.len > 0:
    result.error = answer.get.error
    return
  let response = answer.get.response
  if response.code != 200:
    result.error = response.body
    echo "OPENROUTER ERROR: ", response.body
    return
  try:
    result.reply = response.body.parseReply()
  except CatchableError as err:
    result.error = err.msg
    return
  if result.reply.len == 0:
    result.error = "OpenRouter response did not include content"
    return
  result.ok = true
  echo "AI: ", result.reply

proc last*[T](arr: seq[T], number: int): seq[T] =
  if number >= arr.len: return arr
  return arr[arr.len - number .. ^1]

proc talkToAI*(messages: var seq[ConversationMessage]): string =
  ## Synchronous call to OpenRouter.
  let response = curl.post(
    aiBaseUrl,
    requestHeaders(),
    requestBody(messages),
    timeoutSeconds()
  )
  if response.code != 200:
    echo "OPENROUTER ERROR: ", response.body
    return ""
  let reply = response.body.parseReply()
  echo "AI: ", reply
  messages.add(ConversationMessage(role: "assistant", content: reply))
  reply
