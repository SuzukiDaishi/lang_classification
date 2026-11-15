use anyhow::{Context, Result, bail, ensure};
use half::f16;
use std::{
    collections::HashMap,
    io::{Cursor, Read},
    path::Path,
};

struct Entry {
    ty: u32,
    count: usize,
    offset: usize,
}
pub struct Weights {
    file: Vec<u8>,
    base: usize,
    entries: HashMap<String, Entry>,
    pub labels: Vec<String>,
}
fn take<const N: usize>(c: &mut Cursor<&[u8]>) -> Result<[u8; N]> {
    let mut b = [0; N];
    c.read_exact(&mut b)?;
    Ok(b)
}
fn u32le(c: &mut Cursor<&[u8]>) -> Result<u32> {
    Ok(u32::from_le_bytes(take(c)?))
}
fn size(c: &mut Cursor<&[u8]>) -> Result<usize> {
    Ok(u64::from_le_bytes(take(c)?).try_into()?)
}
fn string(c: &mut Cursor<&[u8]>) -> Result<String> {
    let n = size(c)?;
    ensure!(n <= 65536, "oversized GGUF string");
    let mut b = vec![0; n];
    c.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}
impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::read(path).with_context(|| path.display().to_string())?;
        let mut c = Cursor::new(file.as_slice());
        ensure!(
            take::<4>(&mut c)? == *b"GGUF" && u32le(&mut c)? == 3,
            "expected GGUF v3"
        );
        let n = size(&mut c)?;
        let m = size(&mut c)?;
        ensure!(n <= 1024 && m <= 128, "invalid LID GGUF counts");
        let mut meta = HashMap::new();
        for _ in 0..m {
            let key = string(&mut c)?;
            ensure!(u32le(&mut c)? == 8, "expected string metadata");
            meta.insert(key, string(&mut c)?);
        }
        ensure!(
            meta.get("general.architecture").map(String::as_str) == Some("whisper-lid10"),
            "wrong GGUF architecture"
        );
        let labels: Vec<_> = meta
            .get("lid10.labels")
            .context("missing labels")?
            .split(',')
            .map(String::from)
            .collect();
        ensure!(labels.len() == 99, "expected 99 languages");
        let mut entries = HashMap::new();
        for _ in 0..n {
            let key = string(&mut c)?;
            let dims = u32le(&mut c)?;
            ensure!((1..=4).contains(&dims), "invalid tensor rank");
            let mut count = 1usize;
            for _ in 0..dims {
                count = count
                    .checked_mul(size(&mut c)?)
                    .context("invalid dimensions")?;
            }
            let entry = Entry {
                ty: u32le(&mut c)?,
                count,
                offset: size(&mut c)?,
            };
            ensure!(entries.insert(key, entry).is_none(), "duplicate tensor");
        }
        let base = (c.position() as usize).next_multiple_of(32);
        Ok(Self {
            file,
            entries,
            base,
            labels,
        })
    }
    fn data(&self, name: &str, n: usize) -> Result<(u32, &[u8])> {
        let e = self
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            e.count == n,
            "{name}: expected {n} elements, got {}",
            e.count
        );
        let width = match e.ty {
            0 | 26 => 4,
            1 => 2,
            _ => bail!("unsupported tensor type"),
        };
        let start = self.base.checked_add(e.offset).context("bad offset")?;
        let end = start
            .checked_add(n.checked_mul(width).context("overflow")?)
            .context("bad size")?;
        Ok((
            e.ty,
            self.file.get(start..end).context("truncated GGUF data")?,
        ))
    }
    pub fn values(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let (ty, b) = self.data(name, n)?;
        Ok(match ty {
            0 => b
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
            1 => b
                .chunks_exact(2)
                .map(|x| f16::from_bits(u16::from_le_bytes(x.try_into().unwrap())).to_f32())
                .collect(),
            _ => bail!("expected float tensor {name}"),
        })
    }
    pub fn matrix(&self, name: &str, n: usize) -> Result<Vec<f32>> {
        let blocks = n.div_ceil(60);
        let (ty, bytes) = self.data(&format!("{name}.q10"), blocks * 20)?;
        ensure!(ty == 26, "expected packed 10-bit words");
        let scales = self.values(&format!("{name}.scale"), blocks)?;
        let mut out = Vec::with_capacity(blocks * 60);
        for (block, scale) in bytes.chunks_exact(80).zip(scales) {
            for word in block.chunks_exact(4) {
                let bits = u32::from_le_bytes(word.try_into().unwrap());
                for shift in [0, 10, 20] {
                    out.push((((bits >> shift) & 1023) as i32 - 511) as f32 * scale);
                }
            }
        }
        out.truncate(n);
        Ok(out)
    }
}
