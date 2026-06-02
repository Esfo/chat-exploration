use indexmap::map::Entry;
use indexmap::{IndexMap, IndexSet};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
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

#[derive(Serialize)]
struct Summary {
    paragraphs: usize,
    total_tokens: usize,
    total_words: usize,
    covered_words: usize,
    uncovered_words: usize,
    percent_covered: f64,
    survival_seconds: f64,
    middle_sort_seconds: f64,
    coverage_seconds: f64,
}

fn add_count(map: &mut IndexMap<String, i64>, token: String, count: i64) {
    match map.entry(token) {
        Entry::Occupied(mut entry) => {
            *entry.get_mut() += count;
        }
        Entry::Vacant(entry) => {
            entry.insert(count);
        }
    }
}

fn count_token_chars(text: &[char], token: &[char]) -> i64 {
    if token.is_empty() {
        return 0;
    }

    let mut count = 0;
    let mut index = 0;

    while index + token.len() <= text.len() {
        if &text[index..index + token.len()] == token {
            count += 1;
            index += token.len();
        } else {
            index += 1;
        }
    }

    count
}

fn split_preserve(text: &str, split_tokens: &[Vec<char>]) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut index = 0;

    while index < chars.len() {
        let mut matched_token: Option<&Vec<char>> = None;

        for token in split_tokens {
            if token.is_empty() {
                continue;
            }

            if index + token.len() <= chars.len() && &chars[index..index + token.len()] == token.as_slice() {
                matched_token = Some(token);
                break;
            }
        }

        if let Some(token) = matched_token {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }

            parts.push(token.iter().collect());
            index += token.len();
        } else {
            current.push(chars[index]);
            index += 1;
        }
    }

    if !current.is_empty() {
        parts.push(current);
    }

    parts
}

fn make_token_counter(word: &str) -> IndexMap<String, i64> {
    let chars: Vec<char> = word.chars().collect();
    let mut counter = IndexMap::new();

    for size in 1..=chars.len() {
        for start in 0..=chars.len() - size {
            let token: String = chars[start..start + size].iter().collect();
            add_count(&mut counter, token, 1);
        }
    }

    counter
}

fn find_token_edges(word: &str, token: &str) -> Vec<(usize, usize)> {
    let wordchars: Vec<char> = word.chars().collect();
    let tokenchars: Vec<char> = token.chars().collect();

    let mut edges = Vec::new();

    if tokenchars.is_empty() || tokenchars.len() > wordchars.len() {
        return edges;
    }

    for start in 0..=wordchars.len() - tokenchars.len() {
        if &wordchars[start..start + tokenchars.len()] == tokenchars.as_slice() {
            edges.push((start, start + tokenchars.len()));
        }
    }

    edges
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("usage: token_survival <finaltext.jsonl> <config.json> <outputfolder>");
        std::process::exit(1);
    }

    let finaltextpath = PathBuf::from(&args[1]);
    let configpath = PathBuf::from(&args[2]);
    let outputfolder = PathBuf::from(&args[3]);

    create_dir_all(&outputfolder)?;

    let configfile = File::open(configpath)?;
    let config: Config = serde_json::from_reader(configfile)?;

    let split_tokens: Vec<Vec<char>> = config
        .wordsplits
        .iter()
        .map(|token| token.chars().collect())
        .collect();

    let endpunctuation_tokens: Vec<Vec<char>> = config
        .endpunctuation
        .iter()
        .map(|token| token.chars().collect())
        .collect();

    let inputfile = File::open(finaltextpath)?;
    let reader = BufReader::new(inputfile);

    let mut finaltext = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if line.is_empty() {
            continue;
        }

        let paragraph: String = serde_json::from_str(&line)?;
        finaltext.push(paragraph);
    }

    let nt = Instant::now();
    println!("token survival game begin");

    let mut allwords: IndexSet<String> = IndexSet::new();
    let mut misses: HashMap<String, i64> = HashMap::new();
    let mut survivors: IndexMap<String, i64> = IndexMap::new();
    let mut tokencache: HashMap<String, IndexMap<String, i64>> = HashMap::new();

    for text in &finaltext {
        let words = split_preserve(text, &split_tokens);

        let mut wordcounts: IndexMap<String, i64> = IndexMap::new();

        for word in words {
            add_count(&mut wordcounts, word, 1);
        }

        for word in wordcounts.keys() {
            allwords.insert(word.clone());
        }

        let mut modifications: IndexMap<String, i64> = IndexMap::new();

        let textchars: Vec<char> = text.chars().collect();

        let spacecount = count_token_chars(&textchars, &[' ']);
        add_count(&mut modifications, " ".to_string(), spacecount);

        for punctuation in &endpunctuation_tokens {
            let token: String = punctuation.iter().collect();
            let count = count_token_chars(&textchars, punctuation);
            add_count(&mut modifications, token, count);
        }

        for (word, wordcount) in &wordcounts {
            if !tokencache.contains_key(word) {
                let cached = make_token_counter(word);
                tokencache.insert(word.clone(), cached);
            }

            let cached = tokencache.get(word).unwrap();

            for (token, tokencount) in cached {
                let count = tokencount * wordcount;
                add_count(&mut modifications, token.clone(), count);
            }
        }

        for (token, count) in &modifications {
            add_count(&mut survivors, token.clone(), *count);
        }

        let survivor_snapshot: Vec<String> = survivors.keys().cloned().collect();

        for token in survivor_snapshot {
            if modifications.contains_key(&token) {
                misses.remove(&token);
            } else if let Some(count) = survivors.get_mut(&token) {
                if *count <= 0 {
                    let misscount = misses.entry(token.clone()).or_insert(0);
                    *misscount += 1;

                    if *misscount >= config.survivalrounds {
                        survivors.shift_remove(&token);
                        misses.remove(&token);
                    }
                } else {
                    *count -= 1;
                }
            }
        }
    }

    let survival_seconds = nt.elapsed().as_secs_f64();
    println!("token survival game end {}", survival_seconds);

    let nt = Instant::now();
    println!("middle sort start");

    let mut sortedtokens: Vec<(String, i64, usize)> = survivors
        .iter()
        .enumerate()
        .map(|(index, (token, count))| (token.clone(), *count, index))
        .collect();

    sortedtokens.sort_by(|a, b| {
        let count_order = b.1.cmp(&a.1);

        if count_order == Ordering::Equal {
            a.2.cmp(&b.2)
        } else {
            count_order
        }
    });

    let mut middletokens: Vec<String> = Vec::new();

    for (index, (token, _, _)) in sortedtokens.iter().enumerate() {
        if index % 2 == 1 {
            middletokens.insert(0, token.clone());
        } else {
            middletokens.push(token.clone());
        }
    }

    let middle_sort_seconds = nt.elapsed().as_secs_f64();
    println!("middle sort finished {}", middle_sort_seconds);

    let nt = Instant::now();
    println!("token coverage test start");

    let tokens: Vec<String> = survivors.keys().cloned().collect();

    println!("total tokens: {}", tokens.len());
    println!("total words: {}", allwords.len());

    let mut coveredwords: IndexSet<String> = IndexSet::new();
    let mut uncoveredwords: IndexSet<String> = IndexSet::new();

    for word in &allwords {
        let mut wordedges: HashMap<usize, Vec<(usize, String)>> = HashMap::new();

        for token in &tokens {
            for (start, end) in find_token_edges(word, token) {
                wordedges.entry(start).or_default().push((end, token.clone()));
            }
        }

        let wlen = word.chars().count();
        let mut stack: Vec<(usize, Vec<String>)> = vec![(0, Vec::new())];
        let mut seen: HashSet<usize> = HashSet::new();
        let mut foundcoverage = false;

        while let Some((position, tokenpath)) = stack.pop() {
            if foundcoverage {
                break;
            }

            if position == wlen {
                coveredwords.insert(word.clone());
                foundcoverage = true;
                continue;
            }

            if seen.contains(&position) {
                continue;
            }

            seen.insert(position);

            if let Some(edges) = wordedges.get(&position) {
                for (end, nexttoken) in edges {
                    let mut nextpath = tokenpath.clone();
                    nextpath.push(nexttoken.clone());
                    stack.push((*end, nextpath));
                }
            }
        }

        if !foundcoverage {
            uncoveredwords.insert(word.clone());
        }
    }

    let coverage_seconds = nt.elapsed().as_secs_f64();

    println!("covered words: {} / {}", coveredwords.len(), allwords.len());

    let percent_covered = if allwords.is_empty() {
        0.0
    } else {
        coveredwords.len() as f64 / allwords.len() as f64
    };

    println!("percent covered: {}", percent_covered);
    println!("uncovered words: {}", uncoveredwords.len());
    println!("token coverage test end {}", coverage_seconds);

    let mut survivorsfile = BufWriter::new(File::create(outputfolder.join("survivors.tsv"))?);

    for (token, count) in &survivors {
        writeln!(
            survivorsfile,
            "{}\t{}",
            serde_json::to_string(token)?,
            count
        )?;
    }

    let mut middlefile = BufWriter::new(File::create(outputfolder.join("middle_tokens.jsonl"))?);

    for token in &middletokens {
        writeln!(middlefile, "{}", serde_json::to_string(token)?)?;
    }

    let mut uncoveredfile = BufWriter::new(File::create(outputfolder.join("uncovered_words.jsonl"))?);

    for word in &uncoveredwords {
        writeln!(uncoveredfile, "{}", serde_json::to_string(word)?)?;
    }

    let summary = Summary {
        paragraphs: finaltext.len(),
        total_tokens: tokens.len(),
        total_words: allwords.len(),
        covered_words: coveredwords.len(),
        uncovered_words: uncoveredwords.len(),
        percent_covered,
        survival_seconds,
        middle_sort_seconds,
        coverage_seconds,
    };

    let summaryfile = File::create(outputfolder.join("summary.json"))?;
    serde_json::to_writer_pretty(summaryfile, &summary)?;

    Ok(())
}
