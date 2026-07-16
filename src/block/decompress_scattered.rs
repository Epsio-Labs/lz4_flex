//! Scattered (vectored) block decompression: decode one LZ4 block whose
//! compressed bytes are split across multiple input fragments into multiple
//! output buffers, without gathering the input or staging the output.
//!
//! The hot loop is [`decompress_fragment`], a variant of
//! [`decompress_internal`](super::decompress::decompress_internal) that stops
//! cleanly at buffer edges: a sequence is decoded only if it fits entirely
//! within the current (input fragment, output buffer) pair, and otherwise the
//! positions rewind to the sequence's token and control returns to the
//! driver. The driver advances to the next fragment or buffer, and decodes
//! the one sequence actually straddling an edge byte by byte.
//!
//! Back-references reaching into earlier output buffers reuse the ext-dict
//! mechanism the decoder already has: each output buffer is decoded with the
//! previous (up to) 64 KiB of history as its dictionary, which covers every
//! offset a 16-bit match can encode. When that window lies entirely in the
//! previous buffer it is borrowed in place; only a window spanning several
//! buffers (all smaller than 64 KiB) is assembled into a scratch allocation.

use crate::block::decompress::{
    copy_from_dict, decompress_internal, does_token_fit, duplicate, duplicate_overlapping,
    read_integer_ptr, read_match_offset,
};
use crate::block::{DecompressError, MINMATCH, WINDOW_SIZE};
use crate::fastcpy_unsafe;
use crate::sink::SliceSink;
use alloc::vec::Vec;

/// Why [`decompress_fragment`] stopped.
enum FragmentExit {
    /// The last input fragment was consumed exactly after a literals-only
    /// sequence: the block's end.
    BlockEnd,
    /// The next sequence crosses a fragment or buffer edge. The positions
    /// point at its token, or at the fragment's end when it was consumed
    /// exactly on a sequence boundary.
    Boundary,
}

/// Decompress all bytes of the block spread across `input` fragments into the
/// `output` buffers, filled in order. Returns the number of bytes written,
/// which may be less than the buffers' total capacity.
pub fn decompress_scattered(
    input: &[&[u8]],
    output: &mut [&mut [u8]],
) -> Result<usize, DecompressError> {
    // The common shape, one fragment into one buffer, is exactly a contiguous
    // decode; take the stock path.
    if let ([single_in], [single_out]) = (input, &mut *output) {
        return decompress_internal::<false, _>(single_in, &mut SliceSink::new(single_out, 0), b"");
    }

    let mut frag_idx = next_nonempty(input, 0);
    let mut in_pos = 0usize;
    let mut buf_idx = 0usize;
    let mut out_pos = 0usize;
    let mut window_scratch: Vec<u8> = Vec::new();

    loop {
        let Some(cur_frag) = frag_idx else {
            // Input exhausted without the block-end exit: the final
            // literals-only sequence never materialized.
            return Err(DecompressError::ExpectedAnotherByte);
        };
        // Normalize positions left exactly on an edge (by the byte-wise path
        // or a prior exit): the fragment decoder expects unread input, and a
        // filled buffer with successors should rotate rather than bounce.
        if in_pos == input[cur_frag].len() {
            frag_idx = next_nonempty(input, cur_frag + 1);
            in_pos = 0;
            continue;
        }
        while buf_idx + 1 < output.len() && out_pos == output[buf_idx].len() {
            buf_idx += 1;
            out_pos = 0;
        }
        if buf_idx >= output.len() {
            // Only representable when `output` is empty.
            return Err(DecompressError::OutputTooSmall {
                expected: 1,
                actual: 0,
            });
        }
        let last_input = next_nonempty(input, cur_frag + 1).is_none();
        let (finished, rest) = output.split_at_mut(buf_idx);
        let history: usize = finished.iter().map(|b| b.len()).sum();

        // Specialize away the dict branches whenever no match can reach the
        // history, the same way the stock decoder does with USE_DICT. Only a
        // sequence starting within the first WINDOW_SIZE bytes of a buffer
        // can back-reference past its start (offsets are 16-bit), so the
        // dict-aware variant decodes at most that prefix, through a clipped
        // view of the buffer; everything past it runs dict-free.
        let exit = if history == 0 || out_pos >= WINDOW_SIZE {
            decompress_fragment::<false>(
                input[cur_frag],
                &mut in_pos,
                rest[0],
                &mut out_pos,
                &[],
                last_input,
            )?
        } else {
            let dict = build_dict(finished, &mut window_scratch, history);
            let clip = rest[0].len().min(WINDOW_SIZE);
            decompress_fragment::<true>(
                input[cur_frag],
                &mut in_pos,
                &mut rest[0][..clip],
                &mut out_pos,
                dict,
                last_input,
            )?
        };
        match exit {
            FragmentExit::BlockEnd => return Ok(history + out_pos),
            FragmentExit::Boundary => {
                // A clean stop on a fragment edge, a buffer edge, or the
                // dict-phase clip (which the phase pick above advances past):
                // just re-enter.
                if in_pos == input[cur_frag].len()
                    || out_pos == WINDOW_SIZE
                    || (out_pos == rest[0].len() && buf_idx + 1 < output.len())
                {
                    continue;
                }
                // A sequence truly straddles an edge; decode it byte-wise.
                let done = straddling_sequence(
                    input,
                    &mut frag_idx,
                    &mut in_pos,
                    output,
                    &mut buf_idx,
                    &mut out_pos,
                )?;
                if done {
                    let written: usize =
                        output[..buf_idx].iter().map(|b| b.len()).sum::<usize>() + out_pos;
                    return Ok(written);
                }
            }
        }
    }
}

/// The index of the first non-empty fragment at or after `from`.
fn next_nonempty(input: &[&[u8]], from: usize) -> Option<usize> {
    (from..input.len()).find(|&i| !input[i].is_empty())
}

/// The (up to) [`WINDOW_SIZE`] bytes of history preceding the current output
/// buffer, as the ext-dict for its decode: borrowed straight from the previous
/// buffer when it holds the whole window, otherwise assembled into `scratch`.
fn build_dict<'a>(
    finished: &'a [&'a mut [u8]],
    scratch: &'a mut Vec<u8>,
    history: usize,
) -> &'a [u8] {
    let window = history.min(WINDOW_SIZE);
    if window == 0 {
        return &[];
    }
    let last: &[u8] = finished.last().expect("history is nonzero");
    if last.len() >= window {
        return &last[last.len() - window..];
    }
    scratch.clear();
    scratch.reserve(window);
    // Walk back `window` bytes from the end of the history to find where the
    // window starts, then collect from there forward.
    let mut start_buf = finished.len();
    let mut start_off = 0;
    let mut remaining = window;
    while remaining > 0 {
        start_buf -= 1;
        let len = finished[start_buf].len();
        if len >= remaining {
            start_off = len - remaining;
            remaining = 0;
        } else {
            remaining -= len;
        }
    }
    scratch.extend_from_slice(&finished[start_buf][start_off..]);
    for buffer in &finished[start_buf + 1..] {
        scratch.extend_from_slice(buffer);
    }
    debug_assert_eq!(scratch.len(), window);
    scratch
}

/// Byte-wise cursor over the scattered input, advancing across fragments.
struct ScatteredReader<'a> {
    input: &'a [&'a [u8]],
    frag_idx: &'a mut Option<usize>,
    in_pos: &'a mut usize,
}

impl ScatteredReader<'_> {
    fn read_byte(&mut self) -> Result<u8, DecompressError> {
        loop {
            let Some(i) = *self.frag_idx else {
                return Err(DecompressError::ExpectedAnotherByte);
            };
            if *self.in_pos < self.input[i].len() {
                let byte = self.input[i][*self.in_pos];
                *self.in_pos += 1;
                return Ok(byte);
            }
            *self.frag_idx = next_nonempty(self.input, i + 1);
            *self.in_pos = 0;
        }
    }

    /// Sum a length extension: bytes each adding 255, ending at the first
    /// non-255 byte, which adds itself.
    fn read_extension(&mut self) -> Result<usize, DecompressError> {
        let mut len = 0usize;
        loop {
            let byte = self.read_byte()?;
            len += byte as usize;
            if byte != 255 {
                return Ok(len);
            }
        }
    }

    fn exhausted(&self) -> bool {
        match *self.frag_idx {
            None => true,
            Some(i) => {
                *self.in_pos == self.input[i].len() && next_nonempty(self.input, i + 1).is_none()
            }
        }
    }
}

/// Byte-wise cursor over the scattered output, advancing across buffers, with
/// reads back into everything written so far.
struct ScatteredWriter<'a, 'b> {
    output: &'a mut [&'b mut [u8]],
    buf_idx: &'a mut usize,
    out_pos: &'a mut usize,
}

impl ScatteredWriter<'_, '_> {
    fn write_byte(&mut self, byte: u8) -> Result<(), DecompressError> {
        while *self.buf_idx < self.output.len() && *self.out_pos == self.output[*self.buf_idx].len()
        {
            *self.buf_idx += 1;
            *self.out_pos = 0;
        }
        if *self.buf_idx >= self.output.len() {
            let capacity: usize = self.output.iter().map(|b| b.len()).sum();
            return Err(DecompressError::OutputTooSmall {
                expected: capacity + 1,
                actual: capacity,
            });
        }
        self.output[*self.buf_idx][*self.out_pos] = byte;
        *self.out_pos += 1;
        Ok(())
    }

    fn total_written(&self) -> usize {
        self.output[..*self.buf_idx]
            .iter()
            .map(|b| b.len())
            .sum::<usize>()
            + *self.out_pos
    }

    /// The already-written byte at absolute position `pos`.
    fn read_at(&self, mut pos: usize) -> u8 {
        for buffer in self.output.iter() {
            if pos < buffer.len() {
                return buffer[pos];
            }
            pos -= buffer.len();
        }
        unreachable!("read position precedes total_written");
    }
}

/// Decode exactly one sequence byte by byte across fragment and buffer edges.
/// Returns `true` when it was the block's final, literals-only sequence.
fn straddling_sequence(
    input: &[&[u8]],
    frag_idx: &mut Option<usize>,
    in_pos: &mut usize,
    output: &mut [&mut [u8]],
    buf_idx: &mut usize,
    out_pos: &mut usize,
) -> Result<bool, DecompressError> {
    let mut reader = ScatteredReader {
        input,
        frag_idx,
        in_pos,
    };
    let mut writer = ScatteredWriter {
        output,
        buf_idx,
        out_pos,
    };

    let token = reader.read_byte()?;

    let mut literal_length = (token >> 4) as usize;
    if literal_length == 15 {
        literal_length += reader.read_extension()?;
    }
    for _ in 0..literal_length {
        let byte = reader.read_byte()?;
        writer.write_byte(byte)?;
    }
    if reader.exhausted() {
        return Ok(true);
    }

    let offset = {
        let low = reader.read_byte()?;
        let high = reader.read_byte()?;
        u16::from_le_bytes([low, high]) as usize
    };
    let mut match_length = MINMATCH + (token & 0xF) as usize;
    if match_length == MINMATCH + 15 {
        match_length += reader.read_extension()?;
    }
    if offset == 0 {
        return Err(DecompressError::OffsetZero);
    }
    if offset > writer.total_written() {
        return Err(DecompressError::OffsetOutOfBounds);
    }
    // Re-derive the source position every iteration: an overlapping match
    // (offset < match_length) reads bytes this same loop just produced.
    for _ in 0..match_length {
        let byte = writer.read_at(writer.total_written() - offset);
        writer.write_byte(byte)?;
    }
    Ok(false)
}

/// Run the contiguous decode loop on one (input fragment, output buffer)
/// pair, resuming from `in_pos`/`out_pos`, with `ext_dict` holding the
/// history preceding this buffer.
///
/// Mirrors [`decompress_internal`](super::decompress::decompress_internal)
/// with `USE_DICT` semantics; the difference is the exits. A sequence that
/// would cross the fragment's or buffer's end rewinds to its token and
/// returns [`FragmentExit::Boundary`] instead of erroring, unless
/// `last_input` proves the input is truncated. Only definitive corruption
/// (zero offset, an offset beyond the history) errors here.
fn decompress_fragment<const USE_DICT: bool>(
    input: &[u8],
    in_pos: &mut usize,
    output: &mut [u8],
    out_pos: &mut usize,
    ext_dict: &[u8],
    last_input: bool,
) -> Result<FragmentExit, DecompressError> {
    let ext_dict = if USE_DICT {
        ext_dict
    } else {
        debug_assert!(ext_dict.is_empty());
        &[]
    };
    debug_assert!(*in_pos < input.len());
    debug_assert!(*out_pos <= output.len());

    let output_base = output.as_mut_ptr();
    let output_end = unsafe { output_base.add(output.len()) };
    let mut output_ptr = unsafe { output_base.add(*out_pos) };

    let input_base = input.as_ptr();
    let mut input_ptr = unsafe { input_base.add(*in_pos) };
    let input_ptr_end = unsafe { input_base.add(input.len()) };
    // Margins identical to decompress_internal: enough input for a wild
    // 16-byte literal copy + a match offset + the next token, and enough
    // output for the wild literal and match copies (plus the dict case's
    // slack).
    let safe_distance_from_end = (16 + 2 + 1).min(input.len());
    let input_ptr_safe = unsafe { input_ptr_end.sub(safe_distance_from_end) };
    let mut output_num_safe_bytes = output.len().saturating_sub(16 + 18);
    if USE_DICT {
        // The dict copy can advance the output by up to 17 extra bytes
        // without exiting the hot loop.
        output_num_safe_bytes = output_num_safe_bytes.saturating_sub(17);
    }
    let safe_output_ptr = unsafe { output_base.add(output_num_safe_bytes) };

    loop {
        let seq_input_ptr = input_ptr;
        let seq_output_ptr = output_ptr;
        // Write back the given positions and stop. Rewinding to the sequence
        // start fully undoes it: bytes possibly already copied past
        // `seq_output_ptr` sit beyond the reported position and are
        // rewritten by whoever produces them next.
        macro_rules! exit_at {
            ($in:expr, $out:expr, $exit:expr) => {{
                *in_pos = unsafe { $in.offset_from(input_base) as usize };
                *out_pos = unsafe { $out.offset_from(output_base) as usize };
                return Ok($exit);
            }};
        }
        // In-bounds: the loop is only (re)entered with at least one unread
        // input byte.
        let token = unsafe { input_ptr.read() };
        input_ptr = unsafe { input_ptr.add(1) };

        // Hot path: token needs no length extensions and both sides are far
        // enough from their ends for wild copies. Identical to
        // decompress_internal's.
        if does_token_fit(token)
            && (input_ptr as usize) <= input_ptr_safe as usize
            && output_ptr < safe_output_ptr
        {
            let literal_length = (token >> 4) as usize;
            let mut match_length = MINMATCH + (token & 0xF) as usize;

            unsafe {
                core::ptr::copy_nonoverlapping(input_ptr, output_ptr, 16);
                input_ptr = input_ptr.add(literal_length);
                output_ptr = output_ptr.add(literal_length);
            }

            let offset = read_match_offset(&mut input_ptr)? as usize;

            let output_len = unsafe { output_ptr.offset_from(output_base) as usize };
            if offset > output_len + ext_dict.len() {
                return Err(DecompressError::OffsetOutOfBounds);
            }
            if USE_DICT && offset > output_len {
                let copied = unsafe {
                    copy_from_dict(output_base, &mut output_ptr, ext_dict, offset, match_length)
                };
                if copied == match_length {
                    continue;
                }
                match_length -= copied;
            }

            let start_ptr = unsafe { output_ptr.sub(offset) };
            if offset >= match_length {
                unsafe {
                    core::ptr::copy(start_ptr, output_ptr, 18);
                    output_ptr = output_ptr.add(match_length);
                }
            } else {
                unsafe {
                    duplicate_overlapping(&mut output_ptr, start_ptr, match_length);
                }
            }
            continue;
        }

        // Careful path: exact bounds checks; every shortage that the next
        // fragment or buffer could relieve becomes a Boundary exit.
        macro_rules! boundary {
            () => {
                exit_at!(seq_input_ptr, seq_output_ptr, FragmentExit::Boundary)
            };
        }
        // A mid-sequence shortage of *input* is a buffer edge unless this is
        // the last fragment, where it means the block is truncated.
        macro_rules! boundary_or_err {
            ($err:expr) => {{
                if last_input {
                    return Err($err);
                }
                boundary!();
            }};
        }
        let mut literal_length = (token >> 4) as usize;
        if literal_length != 0 {
            if literal_length == 15 {
                literal_length += match read_integer_ptr(&mut input_ptr, input_ptr_end) {
                    Ok(extension) => extension,
                    Err(err) => boundary_or_err!(err),
                };
            }
            if literal_length > input_ptr_end as usize - input_ptr as usize {
                boundary_or_err!(DecompressError::LiteralOutOfBounds);
            }
            if literal_length > unsafe { output_end.offset_from(output_ptr) as usize } {
                // The literal run continues in the next output buffer.
                boundary!();
            }
            unsafe {
                fastcpy_unsafe::slice_copy(input_ptr, output_ptr, literal_length);
                output_ptr = output_ptr.add(literal_length);
                input_ptr = input_ptr.add(literal_length);
            }
        }

        if input_ptr >= input_ptr_end {
            if last_input {
                // The block's mandated literals-only final sequence.
                exit_at!(input_ptr, output_ptr, FragmentExit::BlockEnd);
            }
            // The match header continues in the next fragment; redo the whole
            // sequence across the edge.
            boundary!();
        }

        if (input_ptr_end as usize) - (input_ptr as usize) < 2 {
            boundary_or_err!(DecompressError::ExpectedAnotherByte);
        }
        let offset = read_match_offset(&mut input_ptr)? as usize;
        let mut match_length = MINMATCH + (token & 0xF) as usize;
        if match_length == MINMATCH + 15 {
            match_length += match read_integer_ptr(&mut input_ptr, input_ptr_end) {
                Ok(extension) => extension,
                Err(err) => boundary_or_err!(err),
            };
        }

        let output_len = unsafe { output_ptr.offset_from(output_base) as usize };
        if offset > output_len + ext_dict.len() {
            return Err(DecompressError::OffsetOutOfBounds);
        }
        if match_length > unsafe { output_end.offset_from(output_ptr) as usize } {
            // The match's output continues in the next buffer.
            boundary!();
        }

        if USE_DICT && offset > output_len {
            let copied = unsafe {
                copy_from_dict(output_base, &mut output_ptr, ext_dict, offset, match_length)
            };
            if copied == match_length {
                if input_ptr >= input_ptr_end {
                    if last_input {
                        return Err(DecompressError::ExpectedAnotherByte);
                    }
                    // Fragment consumed on a sequence boundary.
                    exit_at!(input_ptr, output_ptr, FragmentExit::Boundary);
                }
                continue;
            }
            match_length -= copied;
        }

        let start_ptr = unsafe { output_ptr.sub(offset) };
        unsafe {
            duplicate(&mut output_ptr, output_end, start_ptr, match_length);
        }
        if input_ptr >= input_ptr_end {
            if last_input {
                // A block may not end on a match; the final sequence must be
                // literals-only.
                return Err(DecompressError::ExpectedAnotherByte);
            }
            // Fragment consumed on a sequence boundary.
            exit_at!(input_ptr, output_ptr, FragmentExit::Boundary);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Decompress `compressed` split at the given fragment boundaries into
    /// output buffers of the given sizes, returning the written bytes.
    fn run(compressed: &[u8], input_splits: &[usize], output_sizes: &[usize]) -> Vec<u8> {
        try_run(compressed, input_splits, output_sizes).unwrap()
    }

    fn try_run(
        compressed: &[u8],
        input_splits: &[usize],
        output_sizes: &[usize],
    ) -> Result<Vec<u8>, DecompressError> {
        let mut fragments: Vec<&[u8]> = Vec::new();
        let mut prev = 0;
        for &split in input_splits {
            fragments.push(&compressed[prev..split]);
            prev = split;
        }
        fragments.push(&compressed[prev..]);

        let mut buffers: Vec<Vec<u8>> = output_sizes.iter().map(|&n| vec![0u8; n]).collect();
        let mut output: Vec<&mut [u8]> = buffers.iter_mut().map(|b| b.as_mut_slice()).collect();
        let written = decompress_scattered(&fragments, &mut output)?;
        let mut bytes = buffers.concat();
        bytes.truncate(written);
        Ok(bytes)
    }

    /// Text-like payload with matches at many distances, larger than the
    /// 64 KiB window so far offsets and window-eviction both occur.
    fn payload_200k() -> Vec<u8> {
        let mut state = 0x2545F4914F6CDD1Du64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            state >> 33
        };
        let mut payload = Vec::new();
        while payload.len() < 200 * 1024 {
            let n = next();
            payload.extend_from_slice(
                alloc::format!("/section/{}/entry-{}?tag={:x};", n % 40, n % 3000, n).as_bytes(),
            );
        }
        payload.truncate(200 * 1024);
        payload
    }

    fn small_payload() -> Vec<u8> {
        b"abcabcabc"
            .repeat(50)
            .into_iter()
            .chain(vec![7u8; 300])
            .collect()
    }

    /// Incompressible input compresses to a single giant literal run whose
    /// length rides in a multi-KiB tail of 255-bytes, exercising the
    /// word-at-a-time integer reader on both decode paths.
    #[test]
    fn incompressible_long_length_extensions() {
        let mut state = 0x853C49E6748FEA9Bu64;
        let payload: Vec<u8> = (0..2 * 1024 * 1024)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                (state >> 33) as u8
            })
            .collect();
        let compressed = crate::block::compress(&payload);

        let contiguous = run(&compressed, &[], &[payload.len()]);
        let scattered = run(
            &compressed,
            &[compressed.len() / 3],
            &[payload.len() / 2, payload.len() - payload.len() / 2],
        );

        assert_eq!(contiguous, payload);
        assert_eq!(scattered, payload);
    }

    #[test]
    fn contiguous_matches_stock_decoder() {
        let payload = payload_200k();
        let compressed = crate::block::compress(&payload);

        let out = run(&compressed, &[], &[payload.len()]);

        assert_eq!(out, payload);
    }

    #[test]
    fn every_input_split_small() {
        let payload = small_payload();
        let compressed = crate::block::compress(&payload);

        for split in 0..=compressed.len() {
            let out = run(&compressed, &[split], &[payload.len()]);
            assert_eq!(out, payload, "input split at {split}");
        }
    }

    #[test]
    fn every_output_split_small() {
        let payload = small_payload();
        let compressed = crate::block::compress(&payload);

        for split in 1..payload.len() {
            let out = run(&compressed, &[], &[split, payload.len() - split]);
            assert_eq!(out, payload, "output split at {split}");
        }
    }

    /// Output splits across a large payload: matches reach through the dict
    /// into the previous buffer, including splits inside the 64 KiB window.
    #[test]
    fn output_splits_large_payload() {
        let payload = payload_200k();
        let compressed = crate::block::compress(&payload);

        for split in [1, 100, 4096, 65535, 65536, 65537, 100_000, 199_999] {
            let out = run(&compressed, &[], &[split, payload.len() - split]);
            assert_eq!(out, payload, "output split at {split}");
        }
    }

    /// Small output buffers force the dict window to span several finished
    /// buffers, exercising the assembled-window path.
    #[test]
    fn window_spanning_many_small_buffers() {
        let payload = payload_200k();
        let compressed = crate::block::compress(&payload);
        let mut sizes = vec![10_000; payload.len() / 10_000];
        sizes.push(payload.len() % 10_000);

        let out = run(&compressed, &[], &sizes);

        assert_eq!(out, payload);
    }

    #[test]
    fn scattered_input_and_output_together() {
        let payload = payload_200k();
        let compressed = crate::block::compress(&payload);
        let input_splits: Vec<usize> = (997..compressed.len()).step_by(997).collect();
        let mut sizes = vec![33_333; payload.len() / 33_333];
        sizes.push(payload.len() % 33_333);

        let out = run(&compressed, &input_splits, &sizes);

        assert_eq!(out, payload);
    }

    #[test]
    fn empty_fragments_and_buffers_are_skipped() {
        let payload = small_payload();
        let compressed = crate::block::compress(&payload);
        let fragments: Vec<&[u8]> = vec![&[], &compressed[..7], &[], &compressed[7..], &[]];
        let mut a = vec![0u8; 0];
        let mut b = vec![0u8; 400];
        let mut c = vec![0u8; 0];
        let mut d = vec![0u8; payload.len() - 400];
        let mut output: Vec<&mut [u8]> = vec![&mut a, &mut b, &mut c, &mut d];

        let written = decompress_scattered(&fragments, &mut output).unwrap();

        assert_eq!(written, payload.len());
        assert_eq!([b, d].concat(), payload);
    }

    /// Surplus capacity in the last buffer: decode stops at the block's end
    /// and reports how much was written.
    #[test]
    fn surplus_output_capacity() {
        let payload = small_payload();
        let compressed = crate::block::compress(&payload);

        let single = run(&compressed, &[], &[payload.len() + 4096]);
        let scattered = run(&compressed, &[9], &[256, payload.len()]);

        assert_eq!(single, payload);
        assert_eq!(scattered, payload);
    }

    #[test]
    fn truncated_input_errors() {
        let payload = small_payload();
        let compressed = crate::block::compress(&payload);

        for cut in [1, 5, compressed.len() / 2] {
            let truncated = &compressed[..compressed.len() - cut];
            assert!(
                try_run(truncated, &[], &[payload.len()]).is_err(),
                "cut {cut} contiguous"
            );
            assert!(
                try_run(
                    truncated,
                    &[truncated.len() / 2],
                    &[300, payload.len() - 300]
                )
                .is_err(),
                "cut {cut} scattered"
            );
        }
    }

    #[test]
    fn output_too_small_errors() {
        let payload = small_payload();
        let compressed = crate::block::compress(&payload);

        let contiguous = try_run(&compressed, &[], &[payload.len() - 1]);
        let scattered = try_run(&compressed, &[4], &[100, payload.len() - 101]);

        assert!(contiguous.is_err());
        assert!(scattered.is_err());
    }

    #[test]
    fn offset_before_start_errors() {
        // Token 0x14: 1 literal, then a 4-byte match at offset 9 with only 1
        // byte of history.
        let block = [0x14, b'a', 0x09, 0x00, 0x10, b'b'];

        assert!(matches!(
            try_run(&block, &[], &[8]),
            Err(DecompressError::OffsetOutOfBounds)
        ));
        assert!(try_run(&block, &[3], &[2, 6]).is_err());
    }

    #[test]
    fn offset_zero_errors() {
        let block = [0x14, b'a', 0x00, 0x00, 0x10, b'b'];

        assert!(matches!(
            try_run(&block, &[], &[8]),
            Err(DecompressError::OffsetZero)
        ));
        assert!(try_run(&block, &[3], &[2, 6]).is_err());
    }

    /// Randomized sweep: split points chosen from a deterministic generator,
    /// checked against the stock contiguous decoder.
    #[test]
    fn randomized_splits_match_reference() {
        let payload = payload_200k();
        let compressed = crate::block::compress(&payload);
        let reference = crate::block::decompress(&compressed, payload.len()).unwrap();
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as usize) % bound
        };

        for _ in 0..200 {
            let in_split = 1 + next(compressed.len() - 1);
            let out_split = 1 + next(payload.len() - 1);
            let out = run(
                &compressed,
                &[in_split],
                &[out_split, payload.len() - out_split],
            );
            assert_eq!(out, reference, "in_split {in_split} out_split {out_split}");
        }
    }
}
