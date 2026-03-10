//! M5 starter: deterministic code stub generation from routed coding actions.

use serde::{Deserialize, Serialize};

use super::action::{ActionJson, ActionPayload, ActionType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeGeneration {
    pub language: String,
    pub kind: String,
    pub code: String,
}

pub fn generate_code_from_action(action: &ActionJson, text: &str) -> Option<CodeGeneration> {
    if action.action_type != ActionType::CodingAssist {
        return None;
    }
    let (task, language_hint) = match &action.payload {
        Some(ActionPayload::CodingAssist {
            task,
            language_hint,
        }) => (task.to_ascii_lowercase(), language_hint.to_ascii_lowercase()),
        _ => ("implement".to_string(), "python".to_string()),
    };
    let lower = text.to_ascii_lowercase();
    let language = normalize_language(&language_hint, &lower);
    let code = if lower.contains("binary search") {
        binary_search_stub(&language)
    } else if lower.contains("lru cache") {
        lru_cache_stub(&language)
    } else if lower.contains("dedup") || lower.contains("duplicate") {
        dedup_stub(&language)
    } else if (lower.contains("web server") || lower.contains("http server"))
        && task == "implement"
    {
        web_server_stub(&language)
    } else if lower.contains("topological sort") {
        topological_sort_stub(&language)
    } else if lower.contains("retry") && (lower.contains("test") || lower.contains("pytest") || lower.contains("jest")) {
        retry_test_stub(&language)
    } else if lower.contains("reducer") && lower.contains("functional helper") {
        reducer_refactor_stub(&language)
    } else if lower.contains("state machine") && lower.contains("enum") {
        state_machine_refactor_stub(&language)
    } else if lower.contains("unwrap") && lower.contains("none") {
        unwrap_debug_stub(&language)
    } else if lower.contains("keyerror") || (lower.contains("nested") && lower.contains("dictionary")) {
        keyerror_debug_stub(&language)
    } else if lower.contains("undefined property") || (lower.contains("undefined") && lower.contains("api response")) {
        undefined_property_debug_stub(&language)
    } else if lower.contains("json parsing") && lower.contains("throughput") {
        json_optimize_stub(&language)
    } else if lower.contains("pagination helper") && (lower.contains("pytest") || lower.contains("test")) {
        pagination_test_stub(&language)
    } else if lower.contains("service layer") && lower.contains("business logic") {
        service_refactor_stub(&language)
    } else if lower.contains("deserialization") && lower.contains("allocation") {
        deserialization_optimize_stub(&language)
    } else if lower.contains("dom update") || lower.contains("layout thrashing") {
        dom_optimize_stub(&language)
    } else if lower.contains("debounce") {
        debounce_stub(&language)
    } else if lower.contains("interval merge") || lower.contains("merge") && lower.contains("range")
    {
        interval_merge_stub(&language)
    } else {
        generic_task_stub(&language, &task)
    };
    Some(CodeGeneration {
        language,
        kind: task,
        code,
    })
}

fn normalize_language(hint: &str, text: &str) -> String {
    if hint == "rust" || text.contains(" rust") {
        "rust".to_string()
    } else if hint == "javascript"
        || hint == "typescript"
        || text.contains("javascript")
        || text.contains("typescript")
        || text.contains("node.js")
        || text.contains("nodejs")
    {
        "javascript".to_string()
    } else {
        "python".to_string()
    }
}

fn binary_search_stub(lang: &str) -> String {
    match lang {
        "rust" => "pub fn binary_search(nums: &[i32], target: i32) -> Option<usize> {\n    let mut lo = 0usize;\n    let mut hi = nums.len();\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match nums[mid].cmp(&target) {\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n            std::cmp::Ordering::Equal => return Some(mid),\n        }\n    }\n    None\n}\n".to_string(),
        "javascript" => "function binarySearch(nums, target) {\n  let lo = 0;\n  let hi = nums.length - 1;\n  while (lo <= hi) {\n    const mid = lo + Math.floor((hi - lo) / 2);\n    if (nums[mid] === target) return mid;\n    if (nums[mid] < target) lo = mid + 1;\n    else hi = mid - 1;\n  }\n  return -1;\n}\n".to_string(),
        _ => "def binary_search(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n".to_string(),
    }
}

fn dedup_stub(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::HashSet;\n\npub fn dedup_preserve_order<T: Eq + std::hash::Hash + Clone>(items: &[T]) -> Vec<T> {\n    let mut seen: HashSet<T> = HashSet::new();\n    let mut out = Vec::new();\n    for item in items {\n        if seen.insert(item.clone()) {\n            out.push(item.clone());\n        }\n    }\n    out\n}\n".to_string(),
        "javascript" => "function dedupPreserveOrder(items) {\n  const seen = new Set();\n  const out = [];\n  for (const item of items) {\n    if (!seen.has(item)) {\n      seen.add(item);\n      out.push(item);\n    }\n  }\n  return out;\n}\n".to_string(),
        _ => "def dedup_preserve_order(items: list[str]) -> list[str]:\n    seen = set()\n    out = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            out.append(item)\n    return out\n".to_string(),
    }
}

fn lru_cache_stub(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::{HashMap, VecDeque};\n\npub struct LruCache {\n    cap: usize,\n    map: HashMap<String, String>,\n    order: VecDeque<String>,\n}\n\nimpl LruCache {\n    pub fn new(cap: usize) -> Self {\n        Self { cap, map: HashMap::new(), order: VecDeque::new() }\n    }\n    pub fn get(&mut self, k: &str) -> Option<String> {\n        let v = self.map.get(k)?.clone();\n        self.touch(k.to_string());\n        Some(v)\n    }\n    pub fn put(&mut self, k: String, v: String) {\n        if !self.map.contains_key(&k) && self.map.len() == self.cap {\n            if let Some(old) = self.order.pop_back() {\n                self.map.remove(&old);\n            }\n        }\n        self.map.insert(k.clone(), v);\n        self.touch(k);\n    }\n    fn touch(&mut self, k: String) {\n        self.order.retain(|x| x != &k);\n        self.order.push_front(k);\n    }\n}\n".to_string(),
        "javascript" => "class LruCache {\n  constructor(capacity) {\n    this.capacity = capacity;\n    this.map = new Map();\n  }\n  get(key) {\n    if (!this.map.has(key)) return undefined;\n    const value = this.map.get(key);\n    this.map.delete(key);\n    this.map.set(key, value);\n    return value;\n  }\n  put(key, value) {\n    if (this.map.has(key)) this.map.delete(key);\n    this.map.set(key, value);\n    if (this.map.size > this.capacity) {\n      const oldest = this.map.keys().next().value;\n      this.map.delete(oldest);\n    }\n  }\n}\n".to_string(),
        _ => "from collections import OrderedDict\n\nclass LruCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.data = OrderedDict()\n\n    def get(self, key: str):\n        if key not in self.data:\n            return None\n        self.data.move_to_end(key)\n        return self.data[key]\n\n    def put(self, key: str, value):\n        if key in self.data:\n            self.data.move_to_end(key)\n        self.data[key] = value\n        if len(self.data) > self.capacity:\n            self.data.popitem(last=False)\n".to_string(),
    }
}

fn debounce_stub(lang: &str) -> String {
    if lang == "javascript" {
        "function debounce(fn, waitMs) {\n  let timer = null;\n  return (...args) => {\n    if (timer) clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), waitMs);\n  };\n}\n".to_string()
    } else {
        generic_task_stub(lang, "implement")
    }
}

fn web_server_stub(lang: &str) -> String {
    match lang {
        "rust" => "use std::io::{Read, Write};\nuse std::net::{TcpListener, TcpStream};\n\nfn handle_client(mut stream: TcpStream) {\n    let mut buffer = [0; 1024];\n    let _ = stream.read(&mut buffer);\n    let response = \"HTTP/1.1 200 OK\\r\\nContent-Type: text/plain\\r\\nContent-Length: 13\\r\\n\\r\\nHello, world!\";\n    let _ = stream.write_all(response.as_bytes());\n}\n\npub fn run_server(addr: &str) -> std::io::Result<()> {\n    let listener = TcpListener::bind(addr)?;\n    for stream in listener.incoming() {\n        if let Ok(stream) = stream {\n            handle_client(stream);\n        }\n    }\n    Ok(())\n}\n".to_string(),
        "javascript" => "const http = require('http');\n\nfunction runServer(port = 3000) {\n  const server = http.createServer((req, res) => {\n    res.writeHead(200, { 'Content-Type': 'text/plain' });\n    res.end('Hello, world!');\n  });\n  server.listen(port, () => {\n    console.log(`Server running on http://localhost:${port}`);\n  });\n  return server;\n}\n\nmodule.exports = { runServer };\n".to_string(),
        _ => "from http.server import BaseHTTPRequestHandler, HTTPServer\n\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-Type', 'text/plain')\n        self.end_headers()\n        self.wfile.write(b'Hello, world!')\n\ndef run_server(host: str = '127.0.0.1', port: int = 8000) -> None:\n    server = HTTPServer((host, port), Handler)\n    server.serve_forever()\n".to_string(),
    }
}

fn topological_sort_stub(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::{HashMap, VecDeque};\n\npub fn topo_sort(graph: &HashMap<String, Vec<String>>) -> Vec<String> {\n    let mut indeg: HashMap<String, usize> = HashMap::new();\n    for (u, vs) in graph {\n        indeg.entry(u.clone()).or_insert(0);\n        for v in vs {\n            *indeg.entry(v.clone()).or_insert(0) += 1;\n        }\n    }\n    let mut q: VecDeque<String> = indeg.iter().filter(|(_, d)| **d == 0).map(|(k, _)| k.clone()).collect();\n    let mut out = Vec::new();\n    while let Some(u) = q.pop_front() {\n        out.push(u.clone());\n        if let Some(vs) = graph.get(&u) {\n            for v in vs {\n                if let Some(d) = indeg.get_mut(v) {\n                    *d -= 1;\n                    if *d == 0 {\n                        q.push_back(v.clone());\n                    }\n                }\n            }\n        }\n    }\n    out\n}\n".to_string(),
        "javascript" => "function topoSort(graph) {\n  const indeg = new Map();\n  for (const [u, vs] of Object.entries(graph)) {\n    if (!indeg.has(u)) indeg.set(u, 0);\n    for (const v of vs) indeg.set(v, (indeg.get(v) || 0) + 1);\n  }\n  const q = [...[...indeg.entries()].filter(([, d]) => d === 0).map(([k]) => k)];\n  const out = [];\n  while (q.length) {\n    const u = q.shift();\n    out.push(u);\n    for (const v of graph[u] || []) {\n      indeg.set(v, indeg.get(v) - 1);\n      if (indeg.get(v) === 0) q.push(v);\n    }\n  }\n  return out;\n}\n".to_string(),
        _ => "from collections import deque\n\ndef topo_sort(graph: dict[str, list[str]]) -> list[str]:\n    indeg: dict[str, int] = {}\n    for u, vs in graph.items():\n        indeg.setdefault(u, 0)\n        for v in vs:\n            indeg[v] = indeg.get(v, 0) + 1\n    q = deque([k for k, d in indeg.items() if d == 0])\n    out: list[str] = []\n    while q:\n        u = q.popleft()\n        out.append(u)\n        for v in graph.get(u, []):\n            indeg[v] -= 1\n            if indeg[v] == 0:\n                q.append(v)\n    return out\n".to_string(),
    }
}

fn retry_test_stub(lang: &str) -> String {
    match lang {
        "rust" => "#[cfg(test)]\nmod tests {\n    fn retry_with_backoff<F: FnMut() -> bool>(mut f: F, max: usize) -> bool {\n        for _ in 0..max {\n            if f() {\n                return true;\n            }\n        }\n        false\n    }\n\n    #[test]\n    fn retry_succeeds_before_limit() {\n        let mut attempts = 0;\n        let ok = retry_with_backoff(|| { attempts += 1; attempts >= 3 }, 5);\n        assert!(ok);\n        assert_eq!(attempts, 3);\n    }\n}\n".to_string(),
        "javascript" => "const { describe, it, expect } = require('@jest/globals');\n\ndescribe('retryWithBackoff', () => {\n  it('succeeds before max attempts', async () => {\n    let attempts = 0;\n    const retryWithBackoff = async (fn, max) => {\n      for (let i = 0; i < max; i += 1) {\n        try { return await fn(); } catch (_) {}\n      }\n      throw new Error('failed');\n    };\n    const value = await retryWithBackoff(async () => {\n      attempts += 1;\n      if (attempts < 3) throw new Error('transient');\n      return 42;\n    }, 5);\n    expect(value).toBe(42);\n    expect(attempts).toBe(3);\n  });\n});\n".to_string(),
        _ => "import pytest\n\ndef retry(fn, max_attempts: int):\n    err = None\n    for _ in range(max_attempts):\n        try:\n            return fn()\n        except Exception as e:\n            err = e\n    raise err\n\ndef test_retry_succeeds_before_limit():\n    attempts = {\"n\": 0}\n    def flaky():\n        attempts[\"n\"] += 1\n        if attempts[\"n\"] < 3:\n            raise ValueError(\"transient\")\n        return 42\n    assert retry(flaky, 5) == 42\n    assert attempts[\"n\"] == 3\n".to_string(),
    }
}

fn reducer_refactor_stub(lang: &str) -> String {
    if lang == "javascript" {
        "function addTodo(state, text) {\n  return {\n    ...state,\n    todos: [...state.todos, { id: state.nextId, text, done: false }],\n    nextId: state.nextId + 1,\n  };\n}\n\nfunction toggleTodo(state, id) {\n  return {\n    ...state,\n    todos: state.todos.map((t) => (t.id === id ? { ...t, done: !t.done } : t)),\n  };\n}\n\nfunction reducer(state, action) {\n  switch (action.type) {\n    case 'ADD_TODO':\n      return addTodo(state, action.payload.text);\n    case 'TOGGLE_TODO':\n      return toggleTodo(state, action.payload.id);\n    default:\n      return state;\n  }\n}\n".to_string()
    } else {
        generic_task_stub(lang, "refactor")
    }
}

fn state_machine_refactor_stub(lang: &str) -> String {
    match lang {
        "rust" => "enum State {\n    Init,\n    Running,\n    Stopped,\n}\n\nenum Event {\n    Start,\n    Stop,\n}\n\nfn transition(state: State, event: Event) -> State {\n    match (state, event) {\n        (State::Init, Event::Start) => State::Running,\n        (State::Running, Event::Stop) => State::Stopped,\n        (s, _) => s,\n    }\n}\n".to_string(),
        "javascript" => "const State = { INIT: 'INIT', RUNNING: 'RUNNING', STOPPED: 'STOPPED' };\nconst Event = { START: 'START', STOP: 'STOP' };\n\nfunction transition(state, event) {\n  if (state === State.INIT && event === Event.START) return State.RUNNING;\n  if (state === State.RUNNING && event === Event.STOP) return State.STOPPED;\n  return state;\n}\n".to_string(),
        _ => "from enum import Enum\n\nclass State(Enum):\n    INIT = 'INIT'\n    RUNNING = 'RUNNING'\n    STOPPED = 'STOPPED'\n\nclass Event(Enum):\n    START = 'START'\n    STOP = 'STOP'\n\ndef transition(state: State, event: Event) -> State:\n    if state == State.INIT and event == Event.START:\n        return State.RUNNING\n    if state == State.RUNNING and event == Event.STOP:\n        return State.STOPPED\n    return state\n".to_string(),
    }
}

fn unwrap_debug_stub(lang: &str) -> String {
    if lang == "rust" {
        "fn load_value(opt: Option<String>) -> Result<String, String> {\n    let value = opt.ok_or_else(|| \"missing value\".to_string())?;\n    Ok(value)\n}\n".to_string()
    } else {
        generic_task_stub(lang, "debug")
    }
}

fn keyerror_debug_stub(lang: &str) -> String {
    if lang == "python" {
        "def safe_get_user_id(payload: dict) -> str | None:\n    user = payload.get('user')\n    if not isinstance(user, dict):\n        return None\n    return user.get('id')\n".to_string()
    } else {
        generic_task_stub(lang, "debug")
    }
}

fn undefined_property_debug_stub(lang: &str) -> String {
    if lang == "javascript" {
        "function safeGetUserId(payload) {\n  return payload?.user?.id ?? null;\n}\n".to_string()
    } else {
        generic_task_stub(lang, "debug")
    }
}

fn json_optimize_stub(lang: &str) -> String {
    if lang == "python" {
        "import json\n\ndef parse_batch(lines: list[str]) -> list[dict]:\n    # Reuse local binding for speed in tight loops.\n    loads = json.loads\n    out = []\n    for line in lines:\n        out.append(loads(line))\n    return out\n".to_string()
    } else {
        generic_task_stub(lang, "optimize")
    }
}

fn pagination_test_stub(lang: &str) -> String {
    if lang == "python" {
        "import pytest\n\ndef paginate(items, page, size):\n    start = (page - 1) * size\n    return items[start:start + size]\n\ndef test_paginate_middle_page():\n    items = list(range(20))\n    assert paginate(items, 2, 5) == [5, 6, 7, 8, 9]\n\ndef test_paginate_out_of_range_returns_empty():\n    items = list(range(5))\n    assert paginate(items, 3, 5) == []\n".to_string()
    } else {
        generic_task_stub(lang, "test")
    }
}

fn service_refactor_stub(lang: &str) -> String {
    if lang == "python" {
        "from dataclasses import dataclass\n\n@dataclass\nclass UserService:\n    repo: object\n\n    def fetch_user(self, user_id: str):\n        return self.repo.get_user(user_id)\n\ndef to_user_response(user: dict) -> dict:\n    return {\"id\": user.get(\"id\"), \"name\": user.get(\"name\")}\n".to_string()
    } else {
        generic_task_stub(lang, "refactor")
    }
}

fn deserialization_optimize_stub(lang: &str) -> String {
    if lang == "rust" {
        "use serde::Deserialize;\n\n#[derive(Deserialize)]\nstruct Event<'a> {\n    #[serde(borrow)]\n    kind: &'a str,\n    value: i64,\n}\n\nfn parse_event<'a>(input: &'a str) -> serde_json::Result<Event<'a>> {\n    serde_json::from_str(input)\n}\n".to_string()
    } else {
        generic_task_stub(lang, "optimize")
    }
}

fn dom_optimize_stub(lang: &str) -> String {
    if lang == "javascript" {
        "function renderList(container, items) {\n  const frag = document.createDocumentFragment();\n  for (const item of items) {\n    const li = document.createElement('li');\n    li.textContent = item;\n    frag.appendChild(li);\n  }\n  container.replaceChildren(frag);\n}\n".to_string()
    } else {
        generic_task_stub(lang, "optimize")
    }
}

fn interval_merge_stub(lang: &str) -> String {
    match lang {
        "rust" => "pub fn merge_intervals(mut ranges: Vec<(i32, i32)>) -> Vec<(i32, i32)> {\n    if ranges.is_empty() {\n        return vec![];\n    }\n    ranges.sort_by_key(|r| r.0);\n    let mut out = vec![ranges[0]];\n    for (s, e) in ranges.into_iter().skip(1) {\n        let last = out.last_mut().unwrap();\n        if s <= last.1 {\n            last.1 = last.1.max(e);\n        } else {\n            out.push((s, e));\n        }\n    }\n    out\n}\n".to_string(),
        "javascript" => "function mergeIntervals(ranges) {\n  if (!ranges.length) return [];\n  ranges.sort((a, b) => a[0] - b[0]);\n  const out = [ranges[0].slice()];\n  for (let i = 1; i < ranges.length; i += 1) {\n    const [s, e] = ranges[i];\n    const last = out[out.length - 1];\n    if (s <= last[1]) last[1] = Math.max(last[1], e);\n    else out.push([s, e]);\n  }\n  return out;\n}\n".to_string(),
        _ => "def merge_intervals(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    if not ranges:\n        return []\n    ranges.sort(key=lambda x: x[0])\n    out = [ranges[0]]\n    for s, e in ranges[1:]:\n        ls, le = out[-1]\n        if s <= le:\n            out[-1] = (ls, max(le, e))\n        else:\n            out.append((s, e))\n    return out\n".to_string(),
    }
}

fn generic_task_stub(lang: &str, task: &str) -> String {
    match (lang, task) {
        ("rust", "debug") => "pub fn safe_div(a: f64, b: f64) -> Result<f64, &'static str> {\n    if b == 0.0 {\n        return Err(\"division by zero\");\n    }\n    Ok(a / b)\n}\n".to_string(),
        ("rust", "optimize") => "pub fn count_events(items: &[String]) -> std::collections::HashMap<&str, usize> {\n    let mut counts = std::collections::HashMap::new();\n    for item in items {\n        *counts.entry(item.as_str()).or_insert(0) += 1;\n    }\n    counts\n}\n".to_string(),
        ("rust", "refactor") => "pub struct UserService<R> { repo: R }\n\nimpl<R> UserService<R>\nwhere R: UserRepo {\n    pub fn get_user_name(&self, id: &str) -> Option<String> {\n        self.repo.get_user(id).map(|u| u.name)\n    }\n}\n\npub trait UserRepo {\n    fn get_user(&self, id: &str) -> Option<User>;\n}\n\npub struct User { pub name: String }\n".to_string(),
        ("rust", "test") => "#[cfg(test)]\nmod tests {\n    fn sum(a: i32, b: i32) -> i32 { a + b }\n\n    #[test]\n    fn sum_works() {\n        assert_eq!(sum(2, 3), 5);\n    }\n}\n".to_string(),
        ("rust", _) => "pub fn solve(input: &str) -> String {\n    input.trim().to_string()\n}\n".to_string(),

        ("javascript", "debug") => "function safeDivide(a, b) {\n  if (b === 0) throw new Error('division by zero');\n  return a / b;\n}\n".to_string(),
        ("javascript", "optimize") => "function countEvents(items) {\n  const counts = new Map();\n  for (const item of items) {\n    counts.set(item, (counts.get(item) || 0) + 1);\n  }\n  return counts;\n}\n".to_string(),
        ("javascript", "refactor") => "function toUserResponse(user) {\n  return { id: user.id, name: user.name };\n}\n\nfunction getUser(service, id) {\n  const user = service.fetchUser(id);\n  return user ? toUserResponse(user) : null;\n}\n".to_string(),
        ("javascript", "test") => "const { describe, it, expect } = require('@jest/globals');\n\ndescribe('sum', () => {\n  const sum = (a, b) => a + b;\n  it('adds two numbers', () => {\n    expect(sum(2, 3)).toBe(5);\n  });\n});\n".to_string(),
        ("javascript", _) => "function solve(input) {\n  return String(input).trim();\n}\n".to_string(),

        ("python", "debug") => "def safe_div(a: float, b: float) -> float:\n    if b == 0:\n        raise ValueError('division by zero')\n    return a / b\n".to_string(),
        ("python", "optimize") => "from collections import Counter\n\ndef count_events(items: list[str]) -> dict[str, int]:\n    return dict(Counter(items))\n".to_string(),
        ("python", "refactor") => "from dataclasses import dataclass\n\n@dataclass\nclass User:\n    id: str\n    name: str\n\ndef to_user_response(user: User) -> dict:\n    return {'id': user.id, 'name': user.name}\n".to_string(),
        ("python", "test") => "def sum_values(a: int, b: int) -> int:\n    return a + b\n\ndef test_sum_values() -> None:\n    assert sum_values(2, 3) == 5\n".to_string(),
        (_, _) => "def solve(input_value: str) -> str:\n    return str(input_value).strip()\n".to_string(),
    }
}

