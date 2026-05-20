## OpenRouter-compatible drop-in replacement for bitworld/ais/openai.nim.
##
## Uses the OpenAI-compatible chat completions endpoint provided by OpenRouter.
## Set OPENROUTER_API_KEY in the environment before running.
## Set OPENAI_KEY as alias (openrouter also accepts this for compatibility).

import
  std/[json, options, os, strutils],
  curly, jsony

const
  DefaultOpenAiTimeoutSeconds = 30

var
  aiKey* =
    block:
      let k = getEnv("OPENROUTER_API_KEY")
      if k.len > 0: k else: getEnv("OPENAI_KEY")

let
  aiChatUrl = "https://openrouter.ai/api/v1/chat/completions"
  aiModel = "anthropic/claude-haiku-4.5"
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

proc openAiTimeoutSeconds(): int =
  let value = getEnv("OPENAI_TIMEOUT_SECONDS").strip()
  if value.len == 0:
    return DefaultOpenAiTimeoutSeconds
  try:
    max(1, int(parseFloat(value)))
  except ValueError:
    DefaultOpenAiTimeoutSeconds

proc requestBody(messages: openArray[ConversationMessage]): string =
  var msgs: seq[ConversationMessage]
  for m in messages:
    msgs.add(m)
  let req = ChatRequest(model: aiModel, messages: msgs, max_tokens: 256)
  req.toJson()

proc requestHeaders(): HttpHeaders =
  result["Authorization"] = "Bearer " & aiKey
  result["Content-Type"] = "application/json"
  result["HTTP-Referer"] = "https://softmax.com"
  result["X-Title"] = "lessandro-nirvana"

proc parseReply(body: string): string =
  ## Extracts text from OpenAI-compatible chat completions response.
  let data = parseJson(body)
  let choices = data{"choices"}
  if choices.isNil or choices.kind != JArray or choices.len == 0:
    return ""
  let msg = choices[0]{"message"}
  if msg.isNil:
    return ""
  msg{"content"}.getStr()

proc startTalkToAI*(
  messages: openArray[ConversationMessage],
  tag = ""
): bool =
  if aiKey.len == 0:
    return false
  curl.startRequest(
    "POST",
    aiChatUrl,
    requestHeaders(),
    requestBody(messages),
    openAiTimeoutSeconds(),
    tag
  )
  true

proc pollTalkToAI*(): TalkResult =
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
    echo "ERROR: ", response.body
    return
  try:
    result.reply = response.body.parseReply()
  except CatchableError as err:
    result.error = err.msg
    return
  if result.reply.len == 0:
    result.error = "OpenRouter response did not include output text"
    return
  result.ok = true
  echo "AI: ", result.reply

proc last*[T](arr: seq[T], number: int): seq[T] =
  if number >= arr.len:
    return arr
  return arr[arr.len - number .. ^1]

proc talkToAI*(messages: var seq[ConversationMessage]): string =
  let response = curl.post(
    aiChatUrl,
    requestHeaders(),
    requestBody(messages),
    openAiTimeoutSeconds()
  )
  if response.code != 200:
    echo "ERROR: ", response.body
    return
  let reply = response.body.parseReply()
  echo "AI: ", reply
  messages.add(
    ConversationMessage(
      role: "assistant",
      content: reply
    )
  )
  return reply
