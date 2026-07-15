//! Interleaved A/B of the stock contiguous decode vs `decompress_scattered`
//! with one input fragment and one output buffer (the no-scattering shape).
//! Rounds alternate between the two paths so drift and frequency scaling hit
//! both equally; the per-round minimum is the least-noisy estimator.
//!
//! Run: cargo run --release --no-default-features --features std --example scattered_ab

use std::time::Instant;

fn payload() -> Vec<u8> {
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        state >> 33
    };
    let mut payload = Vec::with_capacity(1 << 20);
    while payload.len() < (1 << 20) {
        let line = format!(
            "https://example.com/catalog/{}/item-{}?session={:x}&ref=search\n",
            next() % 500,
            next() % 100_000,
            next()
        );
        payload.extend_from_slice(line.as_bytes());
    }
    payload.truncate(1 << 20);
    payload
}

fn main() {
    let payload = payload();
    let compressed = lz4_flex::block::compress(&payload);
    let mut out = vec![0u8; payload.len()];
    const ITERS: usize = 300;
    const ROUNDS: usize = 21;

    let mut stock_ns = Vec::new();
    let mut scattered_ns = Vec::new();
    for round in 0..ROUNDS {
        if round % 2 == 0 {
            let start = Instant::now();
            for _ in 0..ITERS {
                let n = lz4_flex::block::decompress_into(&compressed, &mut out).unwrap();
                assert_eq!(n, payload.len());
            }
            stock_ns.push(start.elapsed().as_nanos() as u64 / ITERS as u64);
        } else {
            let input = [&compressed[..]];
            let start = Instant::now();
            for _ in 0..ITERS {
                let mut output = [&mut out[..]];
                let n = lz4_flex::block::decompress_scattered(&input, &mut output).unwrap();
                assert_eq!(n, payload.len());
            }
            scattered_ns.push(start.elapsed().as_nanos() as u64 / ITERS as u64);
        }
    }
    stock_ns.sort();
    scattered_ns.sort();
    let (s_min, s_med) = (stock_ns[0], stock_ns[stock_ns.len() / 2]);
    let (c_min, c_med) = (scattered_ns[0], scattered_ns[scattered_ns.len() / 2]);
    println!("stock decompress_into:        min {s_min} ns  median {s_med} ns");
    println!("scattered 1x1:                min {c_min} ns  median {c_med} ns");
    println!(
        "delta (min): {:+.3}%  (median): {:+.3}%",
        (c_min as f64 / s_min as f64 - 1.0) * 100.0,
        (c_med as f64 / s_med as f64 - 1.0) * 100.0
    );
}
