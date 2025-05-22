# LeanExplore

A search engine for Lean 4 statements. This project provides both an API and a local search engine for [LeanExplore](http://www.leanexplore.com).

The current indexed projects are

* Batteries
* Lean
* Mathlib
* PhysLean
* Std

This code is distributed under an Apache License (see [LICENSE](LICENSE)).

### Chatting with LeanExplore

Here is a quick start guide on how to chat with LeanExplore utilizing the API via MCP and ChatGPT 4o. You will need to input your [OpenAI API key](https://platform.openai.com/api-keys) and your [LeanExplore API key](https://www.leanexplore.com/api-keys) when prompted by the tool. It can save these keys for you for future sessions.

Clone:
```
git clone https://github.com/justincasher/lean-explore.git
cd lean-explore
```
Create and activate virtual environment:
```
python3 -m venv .venv && source .venv/bin/activate
# For Windows, use: .venv\Scripts\activate
```
Install package:
```
pip install -e .
```
Run the chat command:
```
leanexplore chat
```
