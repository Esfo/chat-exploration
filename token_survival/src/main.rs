use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Deserialize)]
struct Config {
    survivalrounds: i64,
    wordsplits: Vec<String>,
    endpunctuation: Vec<String>,
}

fn count_nonoverlapping(text: &[u8], needle: &[u8]) -> i64 {
    let nl = needle.len();
    if nl == 0 || nl > text.len() {
        return 0;
    }
    if nl == 1 {
        let b = needle[0];
        return text.iter().filter(|&&x| x == b).count() as i64;
    }
    let mut count = 0i64;
    let mut i = 0usize;
    let n = text.len();
    let first = needle[0];
    while i + nl <= n {
        if text[i] == first && &text[i..i + nl] == needle {
            count += 1;
            i += nl;
        } else {
            i += 1;
        }
    }
    count
}

fn split_with_seps<'a>(
    text: &'a [u8],
    first_byte: &[Vec<u16>; 256],
    splits: &[Vec<u8>],
) -> Vec<&'a [u8]> {
    let mut parts: Vec<&'a [u8]> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let n = text.len();
    while i < n {
        let candidates = &first_byte[text[i] as usize];
        let mut matched = 0usize;
        for &idx in candidates {
            let sp = &splits[idx as usize];
            let sl = sp.len();
            if i + sl <= n && &text[i..i + sl] == sp.as_slice() {
                matched = sl;
                break;
            }
        }
        if matched > 0 {
            if start < i {
                parts.push(&text[start..i]);
            }
            parts.push(&text[i..i + matched]);
            i += matched;
            start = i;
        } else {
            i += 1;
        }
    }
    if start < n {
        parts.push(&text[start..n]);
    }
    parts
}

// Returns (token, count) pairs in Python's nested-loop first-encounter order.
fn make_token_order(word: &str) -> Vec<(String, i64)> {
    let bytes = word.as_bytes();
    let mut boundaries: Vec<usize> = Vec::with_capacity(word.len() + 1);
    for (i, _) in word.char_indices() {
        boundaries.push(i);
    }
    boundaries.push(bytes.len());
    let cp_count = boundaries.len() - 1;

    let mut order: Vec<String> = Vec::new();
    let mut idx: HashMap<String, usize> = HashMap::new();
    let mut counts: Vec<i64> = Vec::new();

    for size in 1..=cp_count {
        for start in 0..=cp_count - size {
            let token_bytes = &bytes[boundaries[start]..boundaries[start + size]];
            // SAFETY: slice lies between char boundaries of valid utf-8
            let token = unsafe { std::str::from_utf8_unchecked(token_bytes) };
            if let Some(&i) = idx.get(token) {
                counts[i] += 1;
            } else {
                let s = token.to_string();
                idx.insert(s.clone(), order.len());
                order.push(s);
                counts.push(1);
            }
        }
    }
    order.into_iter().zip(counts).collect()
}

struct Interner {
    ids: HashMap<String, u32>,
    strs: Vec<String>,
    survivors: Vec<i64>,
    misses: Vec<i64>,
    alive: Vec<bool>,
    positions: Vec<u32>,
    next_position: u32,
}

impl Interner {
    fn new() -> Self {
        Self {
            ids: HashMap::new(),
            strs: Vec::new(),
            survivors: Vec::new(),
            misses: Vec::new(),
            alive: Vec::new(),
            positions: Vec::new(),
            next_position: 0,
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.strs.len() as u32;
        let owned = s.to_string();
        self.ids.insert(owned.clone(), id);
        self.strs.push(owned);
        self.survivors.push(0);
        self.misses.push(0);
        self.alive.push(true);
        self.positions.push(self.next_position);
        self.next_position += 1;
        id
    }

    fn bump_position(&mut self, id: u32) {
        self.positions[id as usize] = self.next_position;
        self.next_position += 1;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("usage: token_survival <finaltext.jsonl> <config.json> <outputfile> [coverage:0|1]");
        std::process::exit(1);
    }

    let finaltextpath = PathBuf::from(&args[1]);
    let configpath = PathBuf::from(&args[2]);
    let outputfile = PathBuf::from(&args[3]);
    //optional 4th arg toggles the word-completion (coverage) test; defaults on
    let run_coverage = args.get(4).map(|s| s != "0").unwrap_or(true);
    if let Some(parent) = outputfile.parent() {
        create_dir_all(parent)?;
    }

    let config: Config = serde_json::from_reader(File::open(configpath)?)?;
    let survivalrounds = config.survivalrounds;

    let splits: Vec<Vec<u8>> = config
        .wordsplits
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let mut first_byte: [Vec<u16>; 256] = std::array::from_fn(|_| Vec::new());
    for (i, sp) in splits.iter().enumerate() {
        if !sp.is_empty() {
            first_byte[sp[0] as usize].push(i as u16);
        }
    }
    let endpunc: Vec<Vec<u8>> = config
        .endpunctuation
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();

    let reader = BufReader::new(File::open(finaltextpath)?);
    let mut finaltext: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let p: String = serde_json::from_str(&line)?;
        finaltext.push(p);
    }

    let nt = Instant::now();
    println!("token survival game begin");

    let mut interner = Interner::new();
    // Python increments survivors[' '] and survivors[p] every paragraph regardless of count
    // (Counter += 0 still inserts the key), so these IDs must come first.
    let space_id = interner.intern(" ");
    let endpunc_ids: Vec<u32> = config
        .endpunctuation
        .iter()
        .map(|s| interner.intern(s))
        .collect();

    let mut tokencache: HashMap<String, Vec<(u32, i64)>> = HashMap::new();
    let mut modifications: Vec<i64> = vec![0; interner.strs.len()];
    let mut was_touched: Vec<bool> = vec![false; interner.strs.len()];
    let mut touched: Vec<u32> = Vec::new();

    let ensure_capacity = |needed: usize, modifications: &mut Vec<i64>, was_touched: &mut Vec<bool>| {
        if modifications.len() < needed {
            modifications.resize(needed, 0);
            was_touched.resize(needed, false);
        }
    };

    for text in &finaltext {
        let tb = text.as_bytes();
        ensure_capacity(interner.strs.len(), &mut modifications, &mut was_touched);

        // spaces — always considered "modified" even when count is 0
        {
            let id = space_id as usize;
            if !was_touched[id] {
                was_touched[id] = true;
                touched.push(space_id);
            }
            modifications[id] += count_nonoverlapping(tb, b" ");
        }
        for (i, p) in endpunc.iter().enumerate() {
            let id = endpunc_ids[i] as usize;
            if !was_touched[id] {
                was_touched[id] = true;
                touched.push(endpunc_ids[i]);
            }
            modifications[id] += count_nonoverlapping(tb, p);
        }

        let parts = split_with_seps(tb, &first_byte, &splits);

        // wordcounts in first-encounter order; whitespace-only parts are excluded so
        // \n, \t, etc. never enter the cache and ' ' is only counted via the explicit
        // survivors[' '] += text.count(' ') above.
        let mut wc_order: Vec<&[u8]> = Vec::new();
        let mut wc_counts: HashMap<&[u8], i64> = HashMap::new();
        for part in parts {
            if part.is_empty() {
                continue;
            }
            // SAFETY: parts are byte ranges of valid utf-8 text at codepoint boundaries
            let part_str = unsafe { std::str::from_utf8_unchecked(part) };
            if part_str.chars().all(|c| c.is_whitespace()) {
                continue;
            }
            match wc_counts.get_mut(part) {
                Some(v) => *v += 1,
                None => {
                    wc_counts.insert(part, 1);
                    wc_order.push(part);
                }
            }
        }

        for word_bytes in wc_order {
            let wordcount = wc_counts[word_bytes];
            // SAFETY: parts are byte ranges of valid utf-8 text, split at codepoint-safe
            // boundaries (UTF-8 self-synchronizing, all split tokens are full codepoints).
            let word_str = unsafe { std::str::from_utf8_unchecked(word_bytes) };

            if !tokencache.contains_key(word_str) {
                let order_counts = make_token_order(word_str);
                let mut interned: Vec<(u32, i64)> = Vec::with_capacity(order_counts.len());
                for (tok, c) in order_counts {
                    let id = interner.intern(&tok);
                    interned.push((id, c));
                }
                tokencache.insert(word_str.to_string(), interned);
                ensure_capacity(interner.strs.len(), &mut modifications, &mut was_touched);
            }

            let entries = tokencache.get(word_str).unwrap();
            for &(id, tokencount) in entries {
                let idu = id as usize;
                if !was_touched[idu] {
                    was_touched[idu] = true;
                    touched.push(id);
                }
                modifications[idu] += tokencount * wordcount;
            }
        }

        // apply modifications to survivors; if a token previously "died" but its
        // substring reappears, resurrect it (matches Python's del+recreate semantics:
        // survivors[token] is reset to the new count, misses is cleared, and the
        // token's dict-insertion position moves to the end for tie-breaking)
        for &id in &touched {
            let idu = id as usize;
            if !interner.alive[idu] {
                interner.alive[idu] = true;
                interner.survivors[idu] = modifications[idu];
                interner.misses[idu] = 0;
                interner.bump_position(id);
            } else {
                interner.survivors[idu] += modifications[idu];
            }
        }

        // cull every alive token
        let n_ids = interner.strs.len();
        for id in 0..n_ids {
            if !interner.alive[id] {
                continue;
            }
            if was_touched[id] {
                interner.misses[id] = 0;
            } else if interner.survivors[id] <= 0 {
                interner.misses[id] += 1;
                if interner.misses[id] >= survivalrounds {
                    interner.alive[id] = false;
                }
            } else {
                interner.survivors[id] -= 1;
            }
        }

        // reset per-paragraph scratch
        for &id in &touched {
            let idu = id as usize;
            modifications[idu] = 0;
            was_touched[idu] = false;
        }
        touched.clear();
    }

    let survival_seconds = nt.elapsed().as_secs_f64();
    println!("token survival game end {}", survival_seconds);

    let alive_count = interner.alive.iter().filter(|a| **a).count();
    println!("total tokens: {}", alive_count);
    println!("total words: {}", tokencache.len());

    let nt = Instant::now();
    println!("middle sort start");

    let mut alive_list: Vec<(u32, i64, u32)> = (0..interner.strs.len() as u32)
        .filter(|&id| interner.alive[id as usize])
        .map(|id| (id, interner.survivors[id as usize], interner.positions[id as usize]))
        .collect();
    // descending count, then ascending current dict-insertion position — matches
    // Python's stable sort over Counter.items() after potential del+reinsert
    alive_list.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

    // middle-sort: odd indices to the left (in reverse), even indices to the right
    let n = alive_list.len();
    let mut odds: Vec<u32> = Vec::with_capacity(n / 2);
    let mut evens: Vec<u32> = Vec::with_capacity(n - n / 2);
    for (i, (id, _, _)) in alive_list.iter().enumerate() {
        if i % 2 == 1 {
            odds.push(*id);
        } else {
            evens.push(*id);
        }
    }
    odds.reverse();
    let mut middle: Vec<u32> = Vec::with_capacity(n);
    middle.extend(odds);
    middle.extend(evens);

    let middle_sort_seconds = nt.elapsed().as_secs_f64();
    println!("middle sort finished {}", middle_sort_seconds);

    if run_coverage {
    let coverage_nt = Instant::now();
    println!("token coverage test start");

    // tokencache keys are exactly the words encountered (Python's `allwords`).
    let allwords: Vec<&String> = tokencache.keys().collect();
    let token_set: HashSet<&str> = interner
        .strs
        .iter()
        .enumerate()
        .filter(|(i, _)| interner.alive[*i])
        .map(|(_, s)| s.as_str())
        .collect();

    println!("total tokens: {}", token_set.len());
    println!("total words: {}", allwords.len());

    let mut covered: usize = 0;
    let mut boundaries: Vec<usize> = Vec::new();
    let mut reached: Vec<bool> = Vec::new();
    for word in &allwords {
        let bytes = word.as_bytes();
        boundaries.clear();
        for (i, _) in word.char_indices() {
            boundaries.push(i);
        }
        boundaries.push(bytes.len());
        let cp = boundaries.len() - 1;
        reached.clear();
        reached.resize(cp + 1, false);
        reached[0] = true;
        for start in 0..cp {
            if !reached[start] {
                continue;
            }
            for end in (start + 1)..=cp {
                // SAFETY: slice between char boundaries of valid utf-8
                let sub = unsafe {
                    std::str::from_utf8_unchecked(&bytes[boundaries[start]..boundaries[end]])
                };
                if token_set.contains(sub) {
                    reached[end] = true;
                }
            }
            if reached[cp] {
                break;
            }
        }
        if reached[cp] {
            covered += 1;
        }
    }

    let coverage_seconds = coverage_nt.elapsed().as_secs_f64();
    let total_words = allwords.len();
    let uncovered = total_words - covered;
    let percent_covered = if total_words == 0 {
        0.0
    } else {
        covered as f64 / total_words as f64
    };
    println!("covered words: {} / {}", covered, total_words);
    println!("percent covered: {}", percent_covered);
    println!("uncovered words: {}", uncovered);
    println!("token coverage test end {}", coverage_seconds);
    }

    let mut out = BufWriter::new(File::create(&outputfile)?);
    for id in &middle {
        let s = &interner.strs[*id as usize];
        writeln!(out, "{}", serde_json::to_string(s)?)?;
    }
    out.flush()?;

    Ok(())
}
