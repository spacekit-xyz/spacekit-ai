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
    let code = match_template(&lower, &language, &task);
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

// ── Template dispatch ─────────────────────────────────────────────────────

struct TemplateRule {
    keywords: &'static [&'static str],
    require_all: bool,
    generate: fn(&str) -> String,
}

const TEMPLATES: &[TemplateRule] = &[
    TemplateRule { keywords: &["sort", "list"],           require_all: true,  generate: sort_list },
    TemplateRule { keywords: &["bubble sort"],            require_all: false, generate: bubble_sort },
    TemplateRule { keywords: &["quick sort", "quicksort"],require_all: false, generate: quicksort },
    TemplateRule { keywords: &["merge sort", "mergesort"],require_all: false, generate: merge_sort },
    TemplateRule { keywords: &["binary search"],          require_all: false, generate: binary_search },
    TemplateRule { keywords: &["fibonacci", "fib"],       require_all: false, generate: fibonacci },
    TemplateRule { keywords: &["factorial"],              require_all: false, generate: factorial },
    TemplateRule { keywords: &["palindrome"],             require_all: false, generate: palindrome },
    TemplateRule { keywords: &["fizzbuzz", "fizz buzz"],  require_all: false, generate: fizzbuzz },
    TemplateRule { keywords: &["reverse", "string"],      require_all: true,  generate: reverse_string },
    TemplateRule { keywords: &["reverse", "list"],        require_all: true,  generate: reverse_list },
    TemplateRule { keywords: &["flatten", "list"],        require_all: true,  generate: flatten_list },
    TemplateRule { keywords: &["flatten", "nested"],      require_all: true,  generate: flatten_list },
    TemplateRule { keywords: &["two sum", "two-sum"],     require_all: false, generate: two_sum },
    TemplateRule { keywords: &["prime", "number"],        require_all: true,  generate: is_prime },
    TemplateRule { keywords: &["gcd", "greatest common"], require_all: false, generate: gcd },
    TemplateRule { keywords: &["anagram"],                require_all: false, generate: is_anagram },
    TemplateRule { keywords: &["linked list"],            require_all: false, generate: linked_list },
    TemplateRule { keywords: &["stack"],                  require_all: false, generate: stack },
    TemplateRule { keywords: &["queue"],                  require_all: false, generate: queue },
    TemplateRule { keywords: &["matrix", "multiply"],     require_all: true,  generate: matrix_multiply },
    TemplateRule { keywords: &["depth first", "dfs"],     require_all: false, generate: dfs },
    TemplateRule { keywords: &["breadth first", "bfs"],   require_all: false, generate: bfs },
    TemplateRule { keywords: &["lru cache"],              require_all: false, generate: lru_cache },
    TemplateRule { keywords: &["dedup", "duplicate"],     require_all: false, generate: dedup },
    TemplateRule { keywords: &["web server", "http server"], require_all: false, generate: web_server },
    TemplateRule { keywords: &["topological sort"],       require_all: false, generate: topological_sort },
    TemplateRule { keywords: &["debounce"],               require_all: false, generate: debounce },
    TemplateRule { keywords: &["interval", "merge"],      require_all: true,  generate: interval_merge },
    TemplateRule { keywords: &["retry"],                  require_all: false, generate: retry_test },
    TemplateRule { keywords: &["state machine", "enum"],  require_all: false, generate: state_machine },
    TemplateRule { keywords: &["calculator"],             require_all: false, generate: calculator },
    TemplateRule { keywords: &["csv", "parse"],           require_all: true,  generate: csv_parser },
    TemplateRule { keywords: &["rate limit", "throttle"], require_all: false, generate: rate_limiter },
    TemplateRule { keywords: &["memoize", "cache", "decorator"], require_all: false, generate: memoize },
    TemplateRule { keywords: &["binary tree", "bst"],     require_all: false, generate: binary_tree },
    TemplateRule { keywords: &["hash map", "hash table", "dictionary"], require_all: false, generate: hash_map },
    TemplateRule { keywords: &["countdown", "timer"],     require_all: false, generate: countdown },
    TemplateRule { keywords: &["hello world"],            require_all: false, generate: hello_world },
];

fn match_template(text: &str, language: &str, task: &str) -> String {
    for rule in TEMPLATES {
        let matched = if rule.require_all {
            rule.keywords.iter().all(|kw| text.contains(kw))
        } else {
            rule.keywords.iter().any(|kw| text.contains(kw))
        };
        if matched {
            return (rule.generate)(language);
        }
    }
    smart_fallback(text, language, task)
}

fn smart_fallback(text: &str, lang: &str, _task: &str) -> String {
    let fn_name = extract_function_name(text);
    let desc = extract_description(text);
    match lang {
        "rust" => format!(
            "/// {desc}\npub fn {fn_name}(input: &str) -> String {{\n    todo!(\"{desc}\")\n}}\n",
        ),
        "javascript" => format!(
            "/**\n * {desc}\n */\nfunction {fn_name}(input) {{\n  throw new Error('TODO: {desc}');\n}}\n",
        ),
        _ => format!(
            "def {fn_name}(input):\n    \"\"\"{desc}\"\"\"\n    raise NotImplementedError(\"{desc}\")\n",
        ),
    }
}

fn extract_function_name(text: &str) -> String {
    let stop = &["a","an","the","that","which","to","for","in","of","with","and","or","my","your","me"];
    let verbs = &["write","create","build","make","implement","generate","code","design","develop","define","craft","produce"];

    let words: Vec<&str> = text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .collect();

    let meaningful: Vec<&str> = words.iter()
        .copied()
        .filter(|w| {
            let l = w.to_ascii_lowercase();
            !stop.contains(&l.as_str()) && !verbs.contains(&l.as_str())
                && !["function","class","method","python","rust","javascript","typescript","program","script","code"].contains(&l.as_str())
        })
        .take(4)
        .collect();

    if meaningful.is_empty() {
        return "solve".to_string();
    }

    meaningful.iter()
        .map(|w| w.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join("_")
}

fn extract_description(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= 80 {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..77])
    }
}

// ── Template implementations ──────────────────────────────────────────────

fn sort_list(lang: &str) -> String {
    match lang {
        "rust" => "pub fn sort_list(items: &mut Vec<i32>) {\n    items.sort();\n}\n\npub fn sort_list_descending(items: &mut Vec<i32>) {\n    items.sort_by(|a, b| b.cmp(a));\n}\n\npub fn sorted_copy(items: &[i32]) -> Vec<i32> {\n    let mut out = items.to_vec();\n    out.sort();\n    out\n}\n".into(),
        "javascript" => "function sortList(items) {\n  return [...items].sort((a, b) => a - b);\n}\n\nfunction sortDescending(items) {\n  return [...items].sort((a, b) => b - a);\n}\n\nfunction sortByKey(items, key) {\n  return [...items].sort((a, b) => {\n    if (a[key] < b[key]) return -1;\n    if (a[key] > b[key]) return 1;\n    return 0;\n  });\n}\n".into(),
        _ => "def sort_list(items: list) -> list:\n    \"\"\"Return a new sorted list (ascending).\"\"\"\n    return sorted(items)\n\n\ndef sort_descending(items: list) -> list:\n    \"\"\"Return a new sorted list (descending).\"\"\"\n    return sorted(items, reverse=True)\n\n\ndef sort_by_key(items: list[dict], key: str) -> list[dict]:\n    \"\"\"Sort a list of dicts by the given key.\"\"\"\n    return sorted(items, key=lambda x: x[key])\n".into(),
    }
}

fn bubble_sort(lang: &str) -> String {
    match lang {
        "rust" => "pub fn bubble_sort(arr: &mut [i32]) {\n    let n = arr.len();\n    for i in 0..n {\n        let mut swapped = false;\n        for j in 0..n - 1 - i {\n            if arr[j] > arr[j + 1] {\n                arr.swap(j, j + 1);\n                swapped = true;\n            }\n        }\n        if !swapped {\n            break;\n        }\n    }\n}\n".into(),
        "javascript" => "function bubbleSort(arr) {\n  const a = [...arr];\n  for (let i = 0; i < a.length; i++) {\n    let swapped = false;\n    for (let j = 0; j < a.length - 1 - i; j++) {\n      if (a[j] > a[j + 1]) {\n        [a[j], a[j + 1]] = [a[j + 1], a[j]];\n        swapped = true;\n      }\n    }\n    if (!swapped) break;\n  }\n  return a;\n}\n".into(),
        _ => "def bubble_sort(arr: list[int]) -> list[int]:\n    a = arr[:]\n    n = len(a)\n    for i in range(n):\n        swapped = False\n        for j in range(n - 1 - i):\n            if a[j] > a[j + 1]:\n                a[j], a[j + 1] = a[j + 1], a[j]\n                swapped = True\n        if not swapped:\n            break\n    return a\n".into(),
    }
}

fn quicksort(lang: &str) -> String {
    match lang {
        "rust" => "pub fn quicksort(arr: &mut [i32]) {\n    if arr.len() <= 1 {\n        return;\n    }\n    let pivot = arr[arr.len() / 2];\n    let mut lo = 0;\n    let mut hi = arr.len() - 1;\n    while lo <= hi {\n        while arr[lo] < pivot { lo += 1; }\n        while arr[hi] > pivot { hi = hi.wrapping_sub(1); }\n        if lo <= hi {\n            arr.swap(lo, hi);\n            lo += 1;\n            hi = hi.wrapping_sub(1);\n        }\n    }\n    if hi < arr.len() { quicksort(&mut arr[..=hi]); }\n    if lo < arr.len() { quicksort(&mut arr[lo..]); }\n}\n".into(),
        "javascript" => "function quicksort(arr) {\n  if (arr.length <= 1) return arr;\n  const pivot = arr[Math.floor(arr.length / 2)];\n  const left = arr.filter(x => x < pivot);\n  const mid = arr.filter(x => x === pivot);\n  const right = arr.filter(x => x > pivot);\n  return [...quicksort(left), ...mid, ...quicksort(right)];\n}\n".into(),
        _ => "def quicksort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n".into(),
    }
}

fn merge_sort(lang: &str) -> String {
    match lang {
        "rust" => "pub fn merge_sort(arr: &[i32]) -> Vec<i32> {\n    if arr.len() <= 1 {\n        return arr.to_vec();\n    }\n    let mid = arr.len() / 2;\n    let left = merge_sort(&arr[..mid]);\n    let right = merge_sort(&arr[mid..]);\n    merge(&left, &right)\n}\n\nfn merge(a: &[i32], b: &[i32]) -> Vec<i32> {\n    let (mut i, mut j) = (0, 0);\n    let mut out = Vec::with_capacity(a.len() + b.len());\n    while i < a.len() && j < b.len() {\n        if a[i] <= b[j] { out.push(a[i]); i += 1; }\n        else { out.push(b[j]); j += 1; }\n    }\n    out.extend_from_slice(&a[i..]);\n    out.extend_from_slice(&b[j..]);\n    out\n}\n".into(),
        "javascript" => "function mergeSort(arr) {\n  if (arr.length <= 1) return arr;\n  const mid = Math.floor(arr.length / 2);\n  const left = mergeSort(arr.slice(0, mid));\n  const right = mergeSort(arr.slice(mid));\n  return merge(left, right);\n}\n\nfunction merge(a, b) {\n  const out = [];\n  let i = 0, j = 0;\n  while (i < a.length && j < b.length) {\n    if (a[i] <= b[j]) out.push(a[i++]);\n    else out.push(b[j++]);\n  }\n  return [...out, ...a.slice(i), ...b.slice(j)];\n}\n".into(),
        _ => "def merge_sort(arr: list[int]) -> list[int]:\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return _merge(left, right)\n\n\ndef _merge(a: list[int], b: list[int]) -> list[int]:\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out\n".into(),
    }
}

fn binary_search(lang: &str) -> String {
    match lang {
        "rust" => "pub fn binary_search(nums: &[i32], target: i32) -> Option<usize> {\n    let mut lo = 0usize;\n    let mut hi = nums.len();\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match nums[mid].cmp(&target) {\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n            std::cmp::Ordering::Equal => return Some(mid),\n        }\n    }\n    None\n}\n".into(),
        "javascript" => "function binarySearch(nums, target) {\n  let lo = 0, hi = nums.length - 1;\n  while (lo <= hi) {\n    const mid = lo + Math.floor((hi - lo) / 2);\n    if (nums[mid] === target) return mid;\n    if (nums[mid] < target) lo = mid + 1;\n    else hi = mid - 1;\n  }\n  return -1;\n}\n".into(),
        _ => "def binary_search(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = lo + (hi - lo) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n".into(),
    }
}

fn fibonacci(lang: &str) -> String {
    match lang {
        "rust" => "pub fn fibonacci(n: u64) -> u64 {\n    if n <= 1 {\n        return n;\n    }\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 2..=n {\n        let tmp = a + b;\n        a = b;\n        b = tmp;\n    }\n    b\n}\n".into(),
        "javascript" => "function fibonacci(n) {\n  if (n <= 1) return n;\n  let a = 0, b = 1;\n  for (let i = 2; i <= n; i++) {\n    [a, b] = [b, a + b];\n  }\n  return b;\n}\n".into(),
        _ => "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number (0-indexed).\"\"\"\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n".into(),
    }
}

fn factorial(lang: &str) -> String {
    match lang {
        "rust" => "pub fn factorial(n: u64) -> u64 {\n    (1..=n).product()\n}\n\npub fn factorial_recursive(n: u64) -> u64 {\n    if n <= 1 { 1 } else { n * factorial_recursive(n - 1) }\n}\n".into(),
        "javascript" => "function factorial(n) {\n  let result = 1;\n  for (let i = 2; i <= n; i++) result *= i;\n  return result;\n}\n".into(),
        _ => "def factorial(n: int) -> int:\n    \"\"\"Return n! (iterative).\"\"\"\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n".into(),
    }
}

fn palindrome(lang: &str) -> String {
    match lang {
        "rust" => "pub fn is_palindrome(s: &str) -> bool {\n    let chars: Vec<char> = s.chars()\n        .filter(|c| c.is_alphanumeric())\n        .map(|c| c.to_ascii_lowercase())\n        .collect();\n    let n = chars.len();\n    for i in 0..n / 2 {\n        if chars[i] != chars[n - 1 - i] {\n            return false;\n        }\n    }\n    true\n}\n".into(),
        "javascript" => "function isPalindrome(s) {\n  const cleaned = s.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();\n  return cleaned === cleaned.split('').reverse().join('');\n}\n".into(),
        _ => "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if a string is a palindrome (ignoring case and non-alphanumeric).\"\"\"\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n".into(),
    }
}

fn fizzbuzz(lang: &str) -> String {
    match lang {
        "rust" => "pub fn fizzbuzz(n: usize) -> Vec<String> {\n    (1..=n)\n        .map(|i| match (i % 3, i % 5) {\n            (0, 0) => \"FizzBuzz\".into(),\n            (0, _) => \"Fizz\".into(),\n            (_, 0) => \"Buzz\".into(),\n            _ => i.to_string(),\n        })\n        .collect()\n}\n".into(),
        "javascript" => "function fizzbuzz(n) {\n  const out = [];\n  for (let i = 1; i <= n; i++) {\n    if (i % 15 === 0) out.push('FizzBuzz');\n    else if (i % 3 === 0) out.push('Fizz');\n    else if (i % 5 === 0) out.push('Buzz');\n    else out.push(String(i));\n  }\n  return out;\n}\n".into(),
        _ => "def fizzbuzz(n: int) -> list[str]:\n    \"\"\"Return FizzBuzz results from 1 to n.\"\"\"\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out\n".into(),
    }
}

fn reverse_string(lang: &str) -> String {
    match lang {
        "rust" => "pub fn reverse_string(s: &str) -> String {\n    s.chars().rev().collect()\n}\n".into(),
        "javascript" => "function reverseString(s) {\n  return s.split('').reverse().join('');\n}\n".into(),
        _ => "def reverse_string(s: str) -> str:\n    \"\"\"Reverse a string.\"\"\"\n    return s[::-1]\n".into(),
    }
}

fn reverse_list(lang: &str) -> String {
    match lang {
        "rust" => "pub fn reverse_list<T: Clone>(items: &[T]) -> Vec<T> {\n    items.iter().rev().cloned().collect()\n}\n".into(),
        "javascript" => "function reverseList(items) {\n  return [...items].reverse();\n}\n".into(),
        _ => "def reverse_list(items: list) -> list:\n    \"\"\"Return a reversed copy of the list.\"\"\"\n    return items[::-1]\n".into(),
    }
}

fn flatten_list(lang: &str) -> String {
    match lang {
        "rust" => "pub fn flatten(nested: &[Vec<i32>]) -> Vec<i32> {\n    nested.iter().flat_map(|v| v.iter().copied()).collect()\n}\n".into(),
        "javascript" => "function flatten(nested) {\n  return nested.flat(Infinity);\n}\n\nfunction flattenRecursive(arr) {\n  const out = [];\n  for (const item of arr) {\n    if (Array.isArray(item)) out.push(...flattenRecursive(item));\n    else out.push(item);\n  }\n  return out;\n}\n".into(),
        _ => "def flatten(nested: list) -> list:\n    \"\"\"Recursively flatten a nested list.\"\"\"\n    out = []\n    for item in nested:\n        if isinstance(item, list):\n            out.extend(flatten(item))\n        else:\n            out.append(item)\n    return out\n".into(),
    }
}

fn two_sum(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::HashMap;\n\npub fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)> {\n    let mut seen: HashMap<i32, usize> = HashMap::new();\n    for (i, &n) in nums.iter().enumerate() {\n        let complement = target - n;\n        if let Some(&j) = seen.get(&complement) {\n            return Some((j, i));\n        }\n        seen.insert(n, i);\n    }\n    None\n}\n".into(),
        "javascript" => "function twoSum(nums, target) {\n  const seen = new Map();\n  for (let i = 0; i < nums.length; i++) {\n    const complement = target - nums[i];\n    if (seen.has(complement)) return [seen.get(complement), i];\n    seen.set(nums[i], i);\n  }\n  return null;\n}\n".into(),
        _ => "def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:\n    \"\"\"Find two indices whose values sum to target.\"\"\"\n    seen: dict[int, int] = {}\n    for i, n in enumerate(nums):\n        complement = target - n\n        if complement in seen:\n            return (seen[complement], i)\n        seen[n] = i\n    return None\n".into(),
    }
}

fn is_prime(lang: &str) -> String {
    match lang {
        "rust" => "pub fn is_prime(n: u64) -> bool {\n    if n < 2 { return false; }\n    if n < 4 { return true; }\n    if n % 2 == 0 || n % 3 == 0 { return false; }\n    let mut i = 5;\n    while i * i <= n {\n        if n % i == 0 || n % (i + 2) == 0 { return false; }\n        i += 6;\n    }\n    true\n}\n".into(),
        "javascript" => "function isPrime(n) {\n  if (n < 2) return false;\n  if (n < 4) return true;\n  if (n % 2 === 0 || n % 3 === 0) return false;\n  for (let i = 5; i * i <= n; i += 6) {\n    if (n % i === 0 || n % (i + 2) === 0) return false;\n  }\n  return true;\n}\n".into(),
        _ => "def is_prime(n: int) -> bool:\n    \"\"\"Check whether n is a prime number.\"\"\"\n    if n < 2:\n        return False\n    if n < 4:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True\n".into(),
    }
}

fn gcd(lang: &str) -> String {
    match lang {
        "rust" => "pub fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let t = b;\n        b = a % b;\n        a = t;\n    }\n    a\n}\n".into(),
        "javascript" => "function gcd(a, b) {\n  while (b !== 0) [a, b] = [b, a % b];\n  return a;\n}\n".into(),
        _ => "def gcd(a: int, b: int) -> int:\n    \"\"\"Compute the greatest common divisor.\"\"\"\n    while b:\n        a, b = b, a % b\n    return a\n".into(),
    }
}

fn is_anagram(lang: &str) -> String {
    match lang {
        "rust" => "pub fn is_anagram(a: &str, b: &str) -> bool {\n    let mut a: Vec<char> = a.to_lowercase().chars().filter(|c| c.is_alphabetic()).collect();\n    let mut b: Vec<char> = b.to_lowercase().chars().filter(|c| c.is_alphabetic()).collect();\n    a.sort();\n    b.sort();\n    a == b\n}\n".into(),
        "javascript" => "function isAnagram(a, b) {\n  const normalize = s => s.toLowerCase().replace(/[^a-z]/g, '').split('').sort().join('');\n  return normalize(a) === normalize(b);\n}\n".into(),
        _ => "from collections import Counter\n\ndef is_anagram(a: str, b: str) -> bool:\n    \"\"\"Check if two strings are anagrams.\"\"\"\n    clean = lambda s: Counter(c.lower() for c in s if c.isalpha())\n    return clean(a) == clean(b)\n".into(),
    }
}

fn linked_list(lang: &str) -> String {
    match lang {
        "rust" => "pub struct ListNode {\n    pub val: i32,\n    pub next: Option<Box<ListNode>>,\n}\n\nimpl ListNode {\n    pub fn new(val: i32) -> Self {\n        Self { val, next: None }\n    }\n\n    pub fn push_front(head: Option<Box<ListNode>>, val: i32) -> Box<ListNode> {\n        Box::new(ListNode { val, next: head })\n    }\n\n    pub fn to_vec(head: &Option<Box<ListNode>>) -> Vec<i32> {\n        let mut out = Vec::new();\n        let mut cur = head;\n        while let Some(node) = cur {\n            out.push(node.val);\n            cur = &node.next;\n        }\n        out\n    }\n}\n".into(),
        "javascript" => "class ListNode {\n  constructor(val, next = null) {\n    this.val = val;\n    this.next = next;\n  }\n}\n\nfunction toArray(head) {\n  const out = [];\n  let cur = head;\n  while (cur) {\n    out.push(cur.val);\n    cur = cur.next;\n  }\n  return out;\n}\n\nfunction fromArray(arr) {\n  let head = null;\n  for (let i = arr.length - 1; i >= 0; i--) {\n    head = new ListNode(arr[i], head);\n  }\n  return head;\n}\n".into(),
        _ => "class ListNode:\n    def __init__(self, val: int, next_node=None):\n        self.val = val\n        self.next = next_node\n\n    def to_list(self) -> list[int]:\n        out, cur = [], self\n        while cur:\n            out.append(cur.val)\n            cur = cur.next\n        return out\n\n    @staticmethod\n    def from_list(items: list[int]) -> 'ListNode | None':\n        head = None\n        for val in reversed(items):\n            head = ListNode(val, head)\n        return head\n".into(),
    }
}

fn stack(lang: &str) -> String {
    match lang {
        "rust" => "pub struct Stack<T> {\n    data: Vec<T>,\n}\n\nimpl<T> Stack<T> {\n    pub fn new() -> Self { Self { data: Vec::new() } }\n    pub fn push(&mut self, item: T) { self.data.push(item); }\n    pub fn pop(&mut self) -> Option<T> { self.data.pop() }\n    pub fn peek(&self) -> Option<&T> { self.data.last() }\n    pub fn is_empty(&self) -> bool { self.data.is_empty() }\n    pub fn len(&self) -> usize { self.data.len() }\n}\n".into(),
        "javascript" => "class Stack {\n  #data = [];\n  push(item) { this.#data.push(item); }\n  pop() { return this.#data.pop(); }\n  peek() { return this.#data.at(-1); }\n  get isEmpty() { return this.#data.length === 0; }\n  get size() { return this.#data.length; }\n}\n".into(),
        _ => "class Stack:\n    def __init__(self):\n        self._data: list = []\n\n    def push(self, item) -> None:\n        self._data.append(item)\n\n    def pop(self):\n        return self._data.pop()\n\n    def peek(self):\n        return self._data[-1] if self._data else None\n\n    @property\n    def is_empty(self) -> bool:\n        return len(self._data) == 0\n\n    def __len__(self) -> int:\n        return len(self._data)\n".into(),
    }
}

fn queue(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::VecDeque;\n\npub struct Queue<T> {\n    data: VecDeque<T>,\n}\n\nimpl<T> Queue<T> {\n    pub fn new() -> Self { Self { data: VecDeque::new() } }\n    pub fn enqueue(&mut self, item: T) { self.data.push_back(item); }\n    pub fn dequeue(&mut self) -> Option<T> { self.data.pop_front() }\n    pub fn peek(&self) -> Option<&T> { self.data.front() }\n    pub fn is_empty(&self) -> bool { self.data.is_empty() }\n}\n".into(),
        "javascript" => "class Queue {\n  #data = [];\n  enqueue(item) { this.#data.push(item); }\n  dequeue() { return this.#data.shift(); }\n  peek() { return this.#data[0]; }\n  get isEmpty() { return this.#data.length === 0; }\n  get size() { return this.#data.length; }\n}\n".into(),
        _ => "from collections import deque\n\nclass Queue:\n    def __init__(self):\n        self._data = deque()\n\n    def enqueue(self, item) -> None:\n        self._data.append(item)\n\n    def dequeue(self):\n        return self._data.popleft()\n\n    def peek(self):\n        return self._data[0] if self._data else None\n\n    @property\n    def is_empty(self) -> bool:\n        return len(self._data) == 0\n".into(),
    }
}

fn matrix_multiply(lang: &str) -> String {
    match lang {
        "rust" => "pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {\n    let rows = a.len();\n    let cols = b[0].len();\n    let inner = b.len();\n    let mut out = vec![vec![0.0; cols]; rows];\n    for i in 0..rows {\n        for k in 0..inner {\n            for j in 0..cols {\n                out[i][j] += a[i][k] * b[k][j];\n            }\n        }\n    }\n    out\n}\n".into(),
        "javascript" => "function matMul(a, b) {\n  const rows = a.length, cols = b[0].length, inner = b.length;\n  const out = Array.from({ length: rows }, () => Array(cols).fill(0));\n  for (let i = 0; i < rows; i++)\n    for (let k = 0; k < inner; k++)\n      for (let j = 0; j < cols; j++)\n        out[i][j] += a[i][k] * b[k][j];\n  return out;\n}\n".into(),
        _ => "def mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n    \"\"\"Multiply two matrices.\"\"\"\n    rows, inner, cols = len(a), len(b), len(b[0])\n    out = [[0.0] * cols for _ in range(rows)]\n    for i in range(rows):\n        for k in range(inner):\n            for j in range(cols):\n                out[i][j] += a[i][k] * b[k][j]\n    return out\n".into(),
    }
}

fn dfs(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::{HashMap, HashSet};\n\npub fn dfs(graph: &HashMap<String, Vec<String>>, start: &str) -> Vec<String> {\n    let mut visited = HashSet::new();\n    let mut order = Vec::new();\n    let mut stack = vec![start.to_string()];\n    while let Some(node) = stack.pop() {\n        if visited.insert(node.clone()) {\n            order.push(node.clone());\n            if let Some(neighbors) = graph.get(&node) {\n                for n in neighbors.iter().rev() {\n                    stack.push(n.clone());\n                }\n            }\n        }\n    }\n    order\n}\n".into(),
        "javascript" => "function dfs(graph, start) {\n  const visited = new Set();\n  const order = [];\n  const stack = [start];\n  while (stack.length) {\n    const node = stack.pop();\n    if (visited.has(node)) continue;\n    visited.add(node);\n    order.push(node);\n    for (const neighbor of (graph[node] || []).reverse()) {\n      stack.push(neighbor);\n    }\n  }\n  return order;\n}\n".into(),
        _ => "def dfs(graph: dict[str, list[str]], start: str) -> list[str]:\n    \"\"\"Iterative depth-first search.\"\"\"\n    visited: set[str] = set()\n    order: list[str] = []\n    stack = [start]\n    while stack:\n        node = stack.pop()\n        if node in visited:\n            continue\n        visited.add(node)\n        order.append(node)\n        for neighbor in reversed(graph.get(node, [])):\n            stack.append(neighbor)\n    return order\n".into(),
    }
}

fn bfs(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::{HashMap, HashSet, VecDeque};\n\npub fn bfs(graph: &HashMap<String, Vec<String>>, start: &str) -> Vec<String> {\n    let mut visited = HashSet::new();\n    let mut order = Vec::new();\n    let mut queue = VecDeque::new();\n    queue.push_back(start.to_string());\n    visited.insert(start.to_string());\n    while let Some(node) = queue.pop_front() {\n        order.push(node.clone());\n        if let Some(neighbors) = graph.get(&node) {\n            for n in neighbors {\n                if visited.insert(n.clone()) {\n                    queue.push_back(n.clone());\n                }\n            }\n        }\n    }\n    order\n}\n".into(),
        "javascript" => "function bfs(graph, start) {\n  const visited = new Set([start]);\n  const order = [];\n  const queue = [start];\n  while (queue.length) {\n    const node = queue.shift();\n    order.push(node);\n    for (const neighbor of graph[node] || []) {\n      if (!visited.has(neighbor)) {\n        visited.add(neighbor);\n        queue.push(neighbor);\n      }\n    }\n  }\n  return order;\n}\n".into(),
        _ => "from collections import deque\n\ndef bfs(graph: dict[str, list[str]], start: str) -> list[str]:\n    \"\"\"Breadth-first search.\"\"\"\n    visited = {start}\n    order: list[str] = []\n    queue = deque([start])\n    while queue:\n        node = queue.popleft()\n        order.append(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                visited.add(neighbor)\n                queue.append(neighbor)\n    return order\n".into(),
    }
}

fn calculator(lang: &str) -> String {
    match lang {
        "rust" => "pub fn calculate(expr: &str) -> Result<f64, String> {\n    let tokens: Vec<&str> = expr.split_whitespace().collect();\n    if tokens.len() != 3 {\n        return Err(\"expected: <num> <op> <num>\".into());\n    }\n    let a: f64 = tokens[0].parse().map_err(|_| \"invalid number\")?;\n    let b: f64 = tokens[2].parse().map_err(|_| \"invalid number\")?;\n    match tokens[1] {\n        \"+\" => Ok(a + b),\n        \"-\" => Ok(a - b),\n        \"*\" => Ok(a * b),\n        \"/\" => { if b == 0.0 { Err(\"division by zero\".into()) } else { Ok(a / b) } }\n        op => Err(format!(\"unknown operator: {op}\")),\n    }\n}\n".into(),
        "javascript" => "function calculate(a, op, b) {\n  switch (op) {\n    case '+': return a + b;\n    case '-': return a - b;\n    case '*': return a * b;\n    case '/': if (b === 0) throw new Error('division by zero'); return a / b;\n    default: throw new Error(`unknown operator: ${op}`);\n  }\n}\n".into(),
        _ => "def calculate(a: float, op: str, b: float) -> float:\n    \"\"\"Simple calculator supporting +, -, *, /.\"\"\"\n    if op == '+':\n        return a + b\n    elif op == '-':\n        return a - b\n    elif op == '*':\n        return a * b\n    elif op == '/':\n        if b == 0:\n            raise ValueError('division by zero')\n        return a / b\n    else:\n        raise ValueError(f'unknown operator: {op}')\n".into(),
    }
}

fn csv_parser(lang: &str) -> String {
    match lang {
        "rust" => "pub fn parse_csv(input: &str) -> Vec<Vec<String>> {\n    input\n        .lines()\n        .map(|line| line.split(',').map(|f| f.trim().to_string()).collect())\n        .collect()\n}\n".into(),
        "javascript" => "function parseCsv(input) {\n  return input.trim().split('\\n').map(line =>\n    line.split(',').map(field => field.trim())\n  );\n}\n".into(),
        _ => "def parse_csv(text: str) -> list[list[str]]:\n    \"\"\"Parse simple CSV text into a list of rows.\"\"\"\n    return [\n        [field.strip() for field in line.split(',')]\n        for line in text.strip().splitlines()\n    ]\n".into(),
    }
}

fn rate_limiter(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::VecDeque;\nuse std::time::Instant;\n\npub struct RateLimiter {\n    window_ms: u128,\n    max_requests: usize,\n    timestamps: VecDeque<Instant>,\n}\n\nimpl RateLimiter {\n    pub fn new(window_ms: u128, max_requests: usize) -> Self {\n        Self { window_ms, max_requests, timestamps: VecDeque::new() }\n    }\n\n    pub fn allow(&mut self) -> bool {\n        let now = Instant::now();\n        while let Some(&front) = self.timestamps.front() {\n            if now.duration_since(front).as_millis() > self.window_ms {\n                self.timestamps.pop_front();\n            } else {\n                break;\n            }\n        }\n        if self.timestamps.len() < self.max_requests {\n            self.timestamps.push_back(now);\n            true\n        } else {\n            false\n        }\n    }\n}\n".into(),
        "javascript" => "class RateLimiter {\n  constructor(windowMs, maxRequests) {\n    this.windowMs = windowMs;\n    this.maxRequests = maxRequests;\n    this.timestamps = [];\n  }\n\n  allow() {\n    const now = Date.now();\n    this.timestamps = this.timestamps.filter(t => now - t < this.windowMs);\n    if (this.timestamps.length < this.maxRequests) {\n      this.timestamps.push(now);\n      return true;\n    }\n    return false;\n  }\n}\n".into(),
        _ => "import time\nfrom collections import deque\n\nclass RateLimiter:\n    def __init__(self, window_s: float, max_requests: int):\n        self.window_s = window_s\n        self.max_requests = max_requests\n        self._timestamps: deque[float] = deque()\n\n    def allow(self) -> bool:\n        now = time.monotonic()\n        while self._timestamps and now - self._timestamps[0] > self.window_s:\n            self._timestamps.popleft()\n        if len(self._timestamps) < self.max_requests:\n            self._timestamps.append(now)\n            return True\n        return False\n".into(),
    }
}

fn memoize(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::HashMap;\n\npub fn memoize<F>(f: F) -> impl FnMut(i64) -> i64\nwhere\n    F: Fn(i64) -> i64,\n{\n    let mut cache: HashMap<i64, i64> = HashMap::new();\n    move |x| {\n        if let Some(&v) = cache.get(&x) {\n            return v;\n        }\n        let v = f(x);\n        cache.insert(x, v);\n        v\n    }\n}\n".into(),
        "javascript" => "function memoize(fn) {\n  const cache = new Map();\n  return function (...args) {\n    const key = JSON.stringify(args);\n    if (cache.has(key)) return cache.get(key);\n    const result = fn.apply(this, args);\n    cache.set(key, result);\n    return result;\n  };\n}\n".into(),
        _ => "from functools import wraps\n\ndef memoize(fn):\n    \"\"\"Simple memoization decorator.\"\"\"\n    cache = {}\n\n    @wraps(fn)\n    def wrapper(*args):\n        if args in cache:\n            return cache[args]\n        result = fn(*args)\n        cache[args] = result\n        return result\n\n    wrapper.cache = cache\n    return wrapper\n".into(),
    }
}

fn binary_tree(lang: &str) -> String {
    match lang {
        "rust" => "pub struct TreeNode {\n    pub val: i32,\n    pub left: Option<Box<TreeNode>>,\n    pub right: Option<Box<TreeNode>>,\n}\n\nimpl TreeNode {\n    pub fn new(val: i32) -> Self {\n        Self { val, left: None, right: None }\n    }\n\n    pub fn insert(&mut self, val: i32) {\n        if val < self.val {\n            match &mut self.left {\n                Some(left) => left.insert(val),\n                None => self.left = Some(Box::new(TreeNode::new(val))),\n            }\n        } else {\n            match &mut self.right {\n                Some(right) => right.insert(val),\n                None => self.right = Some(Box::new(TreeNode::new(val))),\n            }\n        }\n    }\n\n    pub fn contains(&self, val: i32) -> bool {\n        if val == self.val { return true; }\n        if val < self.val { self.left.as_ref().map_or(false, |n| n.contains(val)) }\n        else { self.right.as_ref().map_or(false, |n| n.contains(val)) }\n    }\n}\n".into(),
        "javascript" => "class TreeNode {\n  constructor(val) {\n    this.val = val;\n    this.left = null;\n    this.right = null;\n  }\n\n  insert(val) {\n    if (val < this.val) {\n      if (this.left) this.left.insert(val);\n      else this.left = new TreeNode(val);\n    } else {\n      if (this.right) this.right.insert(val);\n      else this.right = new TreeNode(val);\n    }\n  }\n\n  contains(val) {\n    if (val === this.val) return true;\n    if (val < this.val) return this.left?.contains(val) ?? false;\n    return this.right?.contains(val) ?? false;\n  }\n}\n".into(),
        _ => "class TreeNode:\n    def __init__(self, val: int):\n        self.val = val\n        self.left: TreeNode | None = None\n        self.right: TreeNode | None = None\n\n    def insert(self, val: int) -> None:\n        if val < self.val:\n            if self.left:\n                self.left.insert(val)\n            else:\n                self.left = TreeNode(val)\n        else:\n            if self.right:\n                self.right.insert(val)\n            else:\n                self.right = TreeNode(val)\n\n    def contains(self, val: int) -> bool:\n        if val == self.val:\n            return True\n        if val < self.val:\n            return self.left.contains(val) if self.left else False\n        return self.right.contains(val) if self.right else False\n".into(),
    }
}

fn hash_map(lang: &str) -> String {
    match lang {
        "rust" => "use std::collections::HashMap;\n\nfn demo() {\n    let mut map: HashMap<String, i32> = HashMap::new();\n    map.insert(\"alice\".into(), 42);\n    map.insert(\"bob\".into(), 99);\n\n    if let Some(val) = map.get(\"alice\") {\n        println!(\"alice = {val}\");\n    }\n\n    for (k, v) in &map {\n        println!(\"{k}: {v}\");\n    }\n}\n".into(),
        "javascript" => "const map = new Map();\nmap.set('alice', 42);\nmap.set('bob', 99);\n\nconsole.log(map.get('alice')); // 42\n\nfor (const [key, val] of map) {\n  console.log(`${key}: ${val}`);\n}\n".into(),
        _ => "# Python dict is a built-in hash map\ndata: dict[str, int] = {}\ndata['alice'] = 42\ndata['bob'] = 99\n\nprint(data.get('alice'))  # 42\n\nfor key, val in data.items():\n    print(f'{key}: {val}')\n".into(),
    }
}

fn countdown(lang: &str) -> String {
    match lang {
        "rust" => "use std::thread;\nuse std::time::Duration;\n\npub fn countdown(seconds: u64) {\n    for i in (1..=seconds).rev() {\n        println!(\"{i}...\");\n        thread::sleep(Duration::from_secs(1));\n    }\n    println!(\"Go!\");\n}\n".into(),
        "javascript" => "function countdown(seconds) {\n  return new Promise(resolve => {\n    let remaining = seconds;\n    const id = setInterval(() => {\n      if (remaining <= 0) {\n        clearInterval(id);\n        console.log('Go!');\n        resolve();\n      } else {\n        console.log(`${remaining}...`);\n        remaining--;\n      }\n    }, 1000);\n  });\n}\n".into(),
        _ => "import time\n\ndef countdown(seconds: int) -> None:\n    \"\"\"Print a countdown from `seconds` to Go!\"\"\"\n    for i in range(seconds, 0, -1):\n        print(f'{i}...')\n        time.sleep(1)\n    print('Go!')\n".into(),
    }
}

fn hello_world(lang: &str) -> String {
    match lang {
        "rust" => "fn main() {\n    println!(\"Hello, world!\");\n}\n".into(),
        "javascript" => "console.log('Hello, world!');\n".into(),
        _ => "print('Hello, world!')\n".into(),
    }
}

// Keep old stubs as thin dispatchers for backward compatibility
fn lru_cache(l: &str) -> String {
    match l {
        "rust" => "use std::collections::{HashMap, VecDeque};\n\npub struct LruCache {\n    cap: usize,\n    map: HashMap<String, String>,\n    order: VecDeque<String>,\n}\n\nimpl LruCache {\n    pub fn new(cap: usize) -> Self {\n        Self { cap, map: HashMap::new(), order: VecDeque::new() }\n    }\n    pub fn get(&mut self, k: &str) -> Option<String> {\n        let v = self.map.get(k)?.clone();\n        self.touch(k.to_string());\n        Some(v)\n    }\n    pub fn put(&mut self, k: String, v: String) {\n        if !self.map.contains_key(&k) && self.map.len() == self.cap {\n            if let Some(old) = self.order.pop_back() {\n                self.map.remove(&old);\n            }\n        }\n        self.map.insert(k.clone(), v);\n        self.touch(k);\n    }\n    fn touch(&mut self, k: String) {\n        self.order.retain(|x| x != &k);\n        self.order.push_front(k);\n    }\n}\n".into(),
        "javascript" => "class LruCache {\n  constructor(capacity) {\n    this.capacity = capacity;\n    this.map = new Map();\n  }\n  get(key) {\n    if (!this.map.has(key)) return undefined;\n    const value = this.map.get(key);\n    this.map.delete(key);\n    this.map.set(key, value);\n    return value;\n  }\n  put(key, value) {\n    if (this.map.has(key)) this.map.delete(key);\n    this.map.set(key, value);\n    if (this.map.size > this.capacity) {\n      const oldest = this.map.keys().next().value;\n      this.map.delete(oldest);\n    }\n  }\n}\n".into(),
        _ => "from collections import OrderedDict\n\nclass LruCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.data = OrderedDict()\n\n    def get(self, key: str):\n        if key not in self.data:\n            return None\n        self.data.move_to_end(key)\n        return self.data[key]\n\n    def put(self, key: str, value):\n        if key in self.data:\n            self.data.move_to_end(key)\n        self.data[key] = value\n        if len(self.data) > self.capacity:\n            self.data.popitem(last=False)\n".into(),
    }
}

fn dedup(l: &str) -> String { match l { "rust" => "use std::collections::HashSet;\n\npub fn dedup_preserve_order<T: Eq + std::hash::Hash + Clone>(items: &[T]) -> Vec<T> {\n    let mut seen = HashSet::new();\n    items.iter().filter(|i| seen.insert((*i).clone())).cloned().collect()\n}\n".into(), "javascript" => "function dedupPreserveOrder(items) {\n  return [...new Set(items)];\n}\n".into(), _ => "def dedup_preserve_order(items: list) -> list:\n    seen = set()\n    return [x for x in items if not (x in seen or seen.add(x))]\n".into() } }

fn web_server(l: &str) -> String { match l { "rust" => "use std::io::{Read, Write};\nuse std::net::TcpListener;\n\npub fn run_server(addr: &str) -> std::io::Result<()> {\n    let listener = TcpListener::bind(addr)?;\n    for stream in listener.incoming().flatten() {\n        let mut buf = [0; 1024];\n        let _ = stream.read(&mut buf);\n        let resp = \"HTTP/1.1 200 OK\\r\\nContent-Length: 13\\r\\n\\r\\nHello, world!\";\n        let _ = (&stream).write_all(resp.as_bytes());\n    }\n    Ok(())\n}\n".into(), "javascript" => "const http = require('http');\nhttp.createServer((req, res) => {\n  res.writeHead(200, { 'Content-Type': 'text/plain' });\n  res.end('Hello, world!');\n}).listen(3000, () => console.log('listening on :3000'));\n".into(), _ => "from http.server import BaseHTTPRequestHandler, HTTPServer\n\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.end_headers()\n        self.wfile.write(b'Hello, world!')\n\nHTTPServer(('127.0.0.1', 8000), Handler).serve_forever()\n".into() } }

fn topological_sort(l: &str) -> String { match l { "rust" => "use std::collections::{HashMap, VecDeque};\n\npub fn topo_sort(graph: &HashMap<String, Vec<String>>) -> Vec<String> {\n    let mut indeg: HashMap<&str, usize> = HashMap::new();\n    for (u, vs) in graph { indeg.entry(u).or_insert(0); for v in vs { *indeg.entry(v).or_insert(0) += 1; } }\n    let mut q: VecDeque<String> = indeg.iter().filter(|(_, &d)| d == 0).map(|(&k, _)| k.to_string()).collect();\n    let mut out = Vec::new();\n    while let Some(u) = q.pop_front() { out.push(u.clone()); for v in graph.get(&u).into_iter().flatten() { let d = indeg.get_mut(v.as_str()).unwrap(); *d -= 1; if *d == 0 { q.push_back(v.clone()); } } }\n    out\n}\n".into(), "javascript" => "function topoSort(graph) {\n  const indeg = new Map();\n  for (const [u, vs] of Object.entries(graph)) { if (!indeg.has(u)) indeg.set(u, 0); for (const v of vs) indeg.set(v, (indeg.get(v)||0)+1); }\n  const q = [...indeg].filter(([,d])=>d===0).map(([k])=>k), out = [];\n  while (q.length) { const u = q.shift(); out.push(u); for (const v of graph[u]||[]) { indeg.set(v, indeg.get(v)-1); if (!indeg.get(v)) q.push(v); } }\n  return out;\n}\n".into(), _ => "from collections import deque\n\ndef topo_sort(graph: dict[str, list[str]]) -> list[str]:\n    indeg: dict[str, int] = {}\n    for u, vs in graph.items():\n        indeg.setdefault(u, 0)\n        for v in vs: indeg[v] = indeg.get(v, 0) + 1\n    q = deque(k for k, d in indeg.items() if d == 0)\n    out = []\n    while q:\n        u = q.popleft(); out.append(u)\n        for v in graph.get(u, []):\n            indeg[v] -= 1\n            if indeg[v] == 0: q.append(v)\n    return out\n".into() } }

fn debounce(l: &str) -> String { if l == "javascript" { "function debounce(fn, waitMs) {\n  let timer = null;\n  return (...args) => {\n    if (timer) clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), waitMs);\n  };\n}\n".into() } else { smart_fallback("debounce", l, "implement") } }

fn interval_merge(l: &str) -> String { match l { "rust" => "pub fn merge_intervals(mut ranges: Vec<(i32,i32)>) -> Vec<(i32,i32)> {\n    ranges.sort_by_key(|r| r.0);\n    let mut out = vec![ranges[0]];\n    for (s, e) in ranges.into_iter().skip(1) {\n        let last = out.last_mut().unwrap();\n        if s <= last.1 { last.1 = last.1.max(e); } else { out.push((s,e)); }\n    }\n    out\n}\n".into(), "javascript" => "function mergeIntervals(ranges) {\n  ranges.sort((a,b)=>a[0]-b[0]);\n  const out = [ranges[0].slice()];\n  for (const [s,e] of ranges.slice(1)) {\n    const last = out.at(-1);\n    if (s <= last[1]) last[1] = Math.max(last[1], e);\n    else out.push([s,e]);\n  }\n  return out;\n}\n".into(), _ => "def merge_intervals(ranges: list[tuple[int,int]]) -> list[tuple[int,int]]:\n    ranges.sort()\n    out = [ranges[0]]\n    for s, e in ranges[1:]:\n        if s <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], e))\n        else: out.append((s, e))\n    return out\n".into() } }

fn retry_test(l: &str) -> String { smart_fallback("retry with backoff test", l, "test") }
fn state_machine(l: &str) -> String { smart_fallback("state machine with enum transitions", l, "implement") }

