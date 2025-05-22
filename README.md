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

Repo: https://github.com/justincasher/lean-explore

To run leanexplore chat:

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
