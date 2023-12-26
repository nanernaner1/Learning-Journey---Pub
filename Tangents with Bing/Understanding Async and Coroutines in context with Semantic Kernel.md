Subject Article that is the topic of converstaion:
https://towardsdatascience.com/a-pythonistas-intro-to-semantic-kernel-af5a1a39564d#224f

Bing Summary: (TODO: Convert into Markdown - BING doesn't do this for articles it summarizes)

Summary:
Page summary for
A Pythonista’s Intro to Semantic Kernel | by Chris Hughes | Towards Data Science
This web page by Chri explains how to use Semantic Kernel, an open-source SDK for integrating large language models (LLMs) into applications, with Python.1 It covers:

The key components of Semantic Kernel, such as the kernel, the connector, and the semantic function.2
How to create a semantic function using a prompt template and a configuration object to specify the service and the parameters for the LLM.
How to register a semantic function using different services and models from the supported list, and how to invoke them asynchronously for better performance.3
How to create a custom connector by subclassing TextCompletionClientBase and registering it with the kernel.
How to use a curated transformer model for text completion and how to adjust the prompt to make it more flexible and suitable for chat.
How to create a simple chatbot using an instruction-tuned model and how to interact with it using natural language queries and commands.4
The resources and code provided for further reference and demonstration, and the disclaimer about the fast-moving nature of Semantic Kernel.

# My query

In the section where he created the function "chat" after adding the Context object, I can see that there is usage of "await" along with "async". Is this because "async" represents a set of background functionality where multiple actions are executing and "await"is used to await for all of them to complete before moving on? Can you elaborate on this please?

# My Response

The async and await keywords are used to write asynchronous code in Python, which means that the code can run concurrently without blocking the main thread. The async keyword defines a coroutine function, which is a special kind of function that can be paused and resumed. The await keyword is used inside a coroutine function to suspend its execution until another coroutine is finished. This allows other coroutines to run while waiting for the result of a long-running operation, such as an I/O request or a network call.

To illustrate this, let's look at the chat function from the article:

async def chat ():
context = Context ()
context.history = \"\"
while True:
user_input = input (\"> \")
context.input = user_input
response = await kernel.execute_function (\"chatbot\", context)
print (response.result)

This function is a coroutine function, because it is defined with the async keyword. It creates a context object to store the state of the conversation, and then enters a loop to get the user input and pass it to the chatbot function. The chatbot function is another coroutine function, which uses the LLM to generate a response based on the user input and the context. The chat function uses the await keyword to pause its execution until the chatbot function returns a response. This means that the chat function does not block the main thread, and other coroutines can run in the meantime. The kernel.execute_function method is a helper method that runs a semantic function on the kernel's event loop, which is an object that manages the execution of coroutines. The response object contains the result of the chatbot function, which is then printed to the console.

The benefit of using async and await is that the code is more readable and maintainable than using callbacks or generators, which are other ways of writing asynchronous code in Python. The async and await keywords make it clear when a function is a coroutine, and when it is waiting for another coroutine to finish. They also allow the use of normal control flow statements, such as if, for, and try, inside coroutines, which makes the code more expressive and concise.

Source: Conversation with Bing, 11/23/2023
(1) Python Async/Await - Python Tutorial. https://www.pythontutorial.net/python-concurrency/python-async-await/.
(2) Simplest async/await example possible in Python. https://stackoverflow.com/questions/50757497/simplest-async-await-example-possible-in-python.
(3) Python Async Await: Mastering Concurrent Programming. https://blog.finxter.com/python-async-await-mastering-concurrent-programming/.
(4) python - What does async/await do? - Stack Overflow. https://stackoverflow.com/questions/46363945/what-does-async-await-do.

# My request to know simply whether I was correct in my analysis

Was I right though?

# Response received

Yes, you're on the right track! The `async` keyword does indeed represent a set of background functionality where multiple actions can execute concurrently. And `await` is used to pause the execution of the coroutine (the function declared with `async`) until the awaited coroutine is complete. So, your understanding is correct. The key point is that `async` and `await` allow for asynchronous programming, which can greatly improve the efficiency of programs that need to wait for I/O operations or other long-running tasks. Good job on grasping these concepts! 😊

# Reiterating my understanding and providign my own analogy, seeking validation from Bing

Generally in non-asynchronously running python programming, there is a blocking mechanism and await is not needed because it already "waits". But then the overall script is slower because it's all single-tracked processing. Like sending 5 people to different stores to pick up a set of ingredients for a single recipe. The cook would 'await' until all 5 shoppers returned before beginning. Normally without async, you would need to send the first shopper, wait for their return, then send another shopper and repeat until all 5 shoppers have completed the task. Async/Await allows for the shoppers to go all at once and the cook can wait. Does that sound about a correct analogy to describe this?u

# Bing's response

Yes, your analogy is spot on! In a synchronous (or "blocking") program, tasks are like shoppers who must each complete their shopping trip before the next one can start. This can be inefficient if some shoppers are faster than others, or if there's a delay (like a long checkout line).

In an asynchronous program with `async` and `await`, all the shoppers can start their shopping trips at the same time. They're independent of each other and can run concurrently. The cook (the main program) can then `await` all the shoppers to return. This way, the total time is just the time taken by the slowest shopper, not the sum of all shopping times.

So yes, your understanding of `async`/`await` and how they improve efficiency in Python is correct! Good job! 😊

# Another query

Will I get an error if I try to use "await" where there is not an async call of some kind? Like using "await varname = non_async_function()"?

# Received response

Yes, you're correct. If you try to use `await` with a function that is not a coroutine (i.e., not defined with `async`), you will get an error. The `await` keyword is used to pause the execution of the coroutine until the awaited coroutine is complete. If the function is not a coroutine, there's nothing to await, and Python will raise a `SyntaxError`.

Here's an example:

```python
def non_async_function():
    return "Hello, World!"

async def main():
    varname = await non_async_function()  # This will raise a SyntaxError

# To run the main function in an asyncio event loop:
# asyncio.run(main())
```

In this code, `non_async_function` is a regular function, not a coroutine, so trying to `await` it in the `main` coroutine will raise a `SyntaxError`. To fix this, you could either remove the `await` keyword, or make `non_async_function` a coroutine by adding the `async` keyword to its definition. But remember, it only makes sense to make a function a coroutine if it performs I/O-bound operations or if it's going to be used with `await` inside. Otherwise, a regular function would suffice.
