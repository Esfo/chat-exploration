// Hierarchical token survival — Rust port of development/hierarchical-survival-text-search.py
//
// Everything is built off individual characters. Every unique character is a permanent
// "base layer" token that can never be removed (spaces included). Multi-character tokens
// are grown on top as a tree: a length-2 token grows off the base layer, and a length-(k+1)
// token is grown by adding one character to either end of a surviving length-k token.
//
// Each round (= one paragraph) only the FRONTIER leaves play the survival game: a leaf that
// appears gains +1, a leaf that is absent loses 1 and dies at score 0. An internal node
// (a parent that still has a living child) is frozen — neither +1 nor -1 — until all of its
// children die, at which point it becomes a leaf again. A token can only be born off a parent
// that was already alive coming INTO the round, so branches climb exactly one level per round.
//
// Per round the occurrence counting is done by descending the living tree against the raw
// text (not by enumerating every substring): scan bigrams once, then expand only the branches
// whose parent actually appears, reading each token's children straight off the parent's
// occurrence positions. Dead subtrees are never scanned.
//
// At the very end, one consolidation pass: if a parent's cumulative occurrence count equals a
// child's, the parent only ever occurs inside that child, so the parent is dropped and the
// longer token kept. The base layer is never dropped.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Deserialize;
use std::env;
use std::error::Error;
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const DEATHFLOOR: i64 = 0; // a leaf dies when its survival score drops to this

#[derive(Deserialize)]
struct Config {
    // grace period: how many consecutive rounds a leaf may sit at score 0 (absent) before
    // it dies (matches tokens-by-survival.py). a leaf gains +1 when present and -1 when
    // absent; once its score reaches 0, it gets this many further absences before removal.
    survivalrounds: i64,
    // only needed to rebuild `allwords` for the coverage sanity check; the game itself
    // runs on the raw paragraph text so spaces/punctuation are real, linkable characters.
    wordsplits: Vec<String>,
}

struct Tok {
    score: i64,  // +1/-1 survival score (only leaves change it)
    misses: i64, // consecutive absent rounds accrued while score is at the death floor
    count: i64,  // raw cumulative occurrences; only used for consolidation
    born: u64,   // round index this token was (re)born on
}

// drop the last character
fn drop_last(s: &str) -> &str {
    match s.char_indices().next_back() {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

// drop the first character
fn drop_first(s: &str) -> &str {
    let mut it = s.char_indices();
    it.next();
    match it.next() {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

// byte-level split keeping both words and separators (mirrors Python re.split with a capture
// group, where every non-empty piece — words and the separators between them — is kept).
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("usage: token_survival <finaltext.jsonl|-> <config.json|inline> <outputfile> [coverage:0|1]");
        std::process::exit(1);
    }

    let finaltextpath = PathBuf::from(&args[1]);
    let configarg = &args[2];
    let outputfile = PathBuf::from(&args[3]);
    let run_coverage = args.get(4).map(|s| s != "0").unwrap_or(true);
    if let Some(parent) = outputfile.parent() {
        create_dir_all(parent)?;
    }

    // tiny config: a leading '{' marks inline JSON, otherwise treat it as a path
    let config: Config = if configarg.trim_start().starts_with('{') {
        serde_json::from_str(configarg)?
    } else {
        serde_json::from_reader(File::open(configarg)?)?
    };
    let survivalrounds = config.survivalrounds;

    // word-split table (only for the coverage test's allwords)
    let splits: Vec<Vec<u8>> = config.wordsplits.iter().map(|s| s.as_bytes().to_vec()).collect();
    let mut first_byte: [Vec<u16>; 256] = std::array::from_fn(|_| Vec::new());
    for (i, sp) in splits.iter().enumerate() {
        if !sp.is_empty() {
            first_byte[sp[0] as usize].push(i as u16);
        }
    }

    // "-" reads the (huge) finaltext from stdin so it never has to hit disk
    let reader: Box<dyn BufRead> = if finaltextpath.as_os_str() == "-" {
        Box::new(BufReader::new(std::io::stdin()))
    } else {
        Box::new(BufReader::new(File::open(finaltextpath)?))
    };
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
    println!("hierarchical survival game begin");

    // --- persistent state across rounds ---
    let mut singles: FxHashMap<char, i64> = FxHashMap::default(); // permanent base layer
    let mut tokens: FxHashMap<String, Tok> = FxHashMap::default(); // living multi-char tokens
    let mut leaves: FxHashSet<String> = FxHashSet::default(); // the frontier
    let mut children: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
    let mut parents: FxHashMap<String, Vec<String>> = FxHashMap::default();
    let mut allwords: FxHashSet<String> = FxHashSet::default(); // coverage only

    // scratch reused each round
    let mut chars: Vec<char> = Vec::new();
    let mut present: FxHashMap<String, i64> = FxHashMap::default();

    for (round, text) in finaltext.iter().enumerate() {
        let round = round as u64;

        // words are only needed for the coverage sanity check, not for the game
        if run_coverage {
            for part in split_with_seps(text.as_bytes(), &first_byte, &splits) {
                if part.is_empty() {
                    continue;
                }
                // SAFETY: parts are byte ranges of valid utf-8 split at codepoint boundaries
                let s = unsafe { std::str::from_utf8_unchecked(part) };
                if !allwords.contains(s) {
                    allwords.insert(s.to_string());
                }
            }
        }

        // base layer — count every single character (spaces included)
        for c in text.chars() {
            *singles.entry(c).or_insert(0) += 1;
        }

        // index the paragraph by codepoint so token slicing matches Python's str slicing
        chars.clear();
        chars.extend(text.chars());
        let n = chars.len();

        // A token is "alive coming into this round" iff it exists and was born on an earlier
        // round (born < round). Births happen below, after the descent, so during the descent
        // every entry in `tokens` already satisfies this; in the births loop the born<round
        // check is what stops a parent (re)born this round from spawning a child this round.

        // --- descend the living tree against the text ---
        present.clear();
        let mut current: FxHashMap<String, FxHashSet<usize>> = FxHashMap::default();
        for i in 0..n.saturating_sub(1) {
            let bigram: String = chars[i..i + 2].iter().collect();
            current.entry(bigram).or_default().insert(i);
        }

        let mut level = 2usize;
        while !current.is_empty() {
            let mut nxt: FxHashMap<String, FxHashSet<usize>> = FxHashMap::default();
            for (token, positions) in current.iter() {
                present.insert(token.clone(), positions.len() as i64);

                // only living branches grow; a token that wasn't alive coming in is left as a
                // tip this round (its own birth/scoring still happens below)
                let prevalive = tokens.get(token).map_or(false, |t| t.born < round);
                if !prevalive {
                    continue;
                }

                // the children of this branch are its one-character left/right extensions, read
                // straight off the parent's occurrence positions. a set of start positions
                // dedupes an extension reachable from both its prefix and its suffix parent.
                for &i in positions.iter() {
                    if i + level < n {
                        let ext: String = chars[i..i + level + 1].iter().collect();
                        nxt.entry(ext).or_default().insert(i);
                    }
                    if i >= 1 {
                        let ext: String = chars[i - 1..i + level].iter().collect();
                        nxt.entry(ext).or_default().insert(i - 1);
                    }
                }
            }
            current = nxt;
            level += 1;
        }

        // --- births + raw counts ---
        for (token, &occ) in present.iter() {
            if let Some(t) = tokens.get_mut(token) {
                // already living — accrue raw occurrences (score handled below)
                t.count += occ;
                continue;
            }

            // a brand-new token. length-2 tokens grow straight off the base layer; any longer
            // token only reached `present` by extending a living (prevalive) parent, so it is
            // guaranteed eligible — we just recover which parent(s) to link it to.
            let livingparents: Vec<String> = if token.chars().count() == 2 {
                Vec::new()
            } else {
                let pre = drop_last(token);
                let suf = drop_first(token);
                let mut lp: Vec<String> = Vec::new();
                if tokens.get(pre).map_or(false, |t| t.born < round) {
                    lp.push(pre.to_string());
                }
                if tokens.get(suf).map_or(false, |t| t.born < round) {
                    lp.push(suf.to_string());
                }
                if lp.is_empty() {
                    continue; // safety; shouldn't happen given how `present` is built
                }
                lp
            };

            // born as a fresh leaf; the survival game below gives it its first +1
            tokens.insert(
                token.clone(),
                Tok { score: DEATHFLOOR, misses: 0, count: occ, born: round },
            );
            leaves.insert(token.clone());
            for p in &livingparents {
                children.entry(p.clone()).or_default().insert(token.clone());
                leaves.remove(p); // parent now has a child -> frozen, off the frontier
            }
            parents.insert(token.clone(), livingparents);
        }

        // --- the survival game — only the frontier leaves play it ---
        let leaf_snapshot: Vec<String> = leaves.iter().cloned().collect();
        for token in leaf_snapshot {
            if present.contains_key(&token) {
                if let Some(t) = tokens.get_mut(&token) {
                    t.score += 1;
                    t.misses = 0; // reappeared: reset the grace counter
                }
                continue;
            }
            let dead = {
                let t = tokens.get_mut(&token).expect("leaf must be alive");
                if t.score > DEATHFLOOR {
                    t.score -= 1; // still above the floor: just decay
                    false
                } else {
                    // sitting at the death floor: burn a grace round, die after survivalrounds
                    t.misses += 1;
                    t.misses >= survivalrounds
                }
            };
            if !dead {
                continue;
            }
            // this branch tip is dead — remove it and detach from its parents, re-promoting
            // any parent that has now lost its last child
            leaves.remove(&token);
            tokens.remove(&token);
            if let Some(ps) = parents.remove(&token) {
                for p in ps {
                    if let Some(cs) = children.get_mut(&p) {
                        cs.remove(&token);
                        if cs.is_empty() {
                            children.remove(&p);
                            if tokens.contains_key(&p) {
                                leaves.insert(p);
                            }
                        }
                    }
                }
            }
            children.remove(&token);
        }
    }

    let survival_seconds = nt.elapsed().as_secs_f64();
    println!("hierarchical survival game end {}", survival_seconds);
    println!("base layer (single chars): {}", singles.len());
    println!("surviving multi tokens: {}", tokens.len());

    // --- consolidation ---
    let nt = Instant::now();
    println!("consolidation start");
    let mut removed: FxHashSet<String> = FxHashSet::default();
    for (parent, ptok) in tokens.iter() {
        if let Some(cs) = children.get(parent) {
            for child in cs.iter() {
                if let Some(ctok) = tokens.get(child) {
                    if ptok.count == ctok.count {
                        removed.insert(parent.clone());
                        break;
                    }
                }
            }
        }
    }
    println!("consolidated away: {}", removed.len());
    println!("consolidation end {}", nt.elapsed().as_secs_f64());

    // --- coverage sanity check ---
    if run_coverage {
        let nt = Instant::now();
        println!("token coverage test start");

        // final token set = permanent base layer + surviving grown tokens (minus consolidated)
        let mut token_set: FxHashSet<String> = FxHashSet::default();
        for c in singles.keys() {
            token_set.insert(c.to_string());
        }
        for t in tokens.keys() {
            if !removed.contains(t) {
                token_set.insert(t.clone());
            }
        }

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

        let total_words = allwords.len();
        let uncovered = total_words - covered;
        let percent = if total_words == 0 { 0.0 } else { covered as f64 / total_words as f64 };
        println!("covered words: {} / {}", covered, total_words);
        println!("percent covered: {}", percent);
        println!("uncovered words: {}", uncovered);
        println!("token coverage test end {}", nt.elapsed().as_secs_f64());
    }

    // --- write final tokens: one JSON-encoded string per line (order irrelevant) ---
    let mut out = BufWriter::new(File::create(&outputfile)?);
    for c in singles.keys() {
        let s = c.to_string();
        writeln!(out, "{}", serde_json::to_string(&s)?)?;
    }
    for token in tokens.keys() {
        if removed.contains(token) {
            continue;
        }
        writeln!(out, "{}", serde_json::to_string(token)?)?;
    }
    out.flush()?;

    Ok(())
}
