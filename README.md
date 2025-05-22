# LeanExplore

A search engine for Lean 4 statements. This project provides both an API and a local search engine for LeanExplore. Visit the main website at [www.leanexplore.com](http://www.leanexplore.com).

The current indexed projects are

* Batteries
* Lean
* Mathlib
* PhysLean
* Std

This code is distributed under an Apache License (see [LICENSE](LICENSE)).

### Chatting with LeanExplore

Here is a quick start guide on how to chat with LeanExplore utilizing the API via MCP and ChatGPT 4o.

Clone & cd:
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
