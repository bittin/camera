// SPDX-License-Identifier: GPL-3.0-only

//! Motion Photo 1.0 still-image packaging.
//!
//! Still bytes are preserved, not re-encoded. For AVIF, the existing `meta`
//! becomes a same-size `free` box, and its metadata items are copied into a
//! relocated `meta`; original absolute image/Exif offsets remain unchanged.
//! The XMP MIME item is associated with the primary image by a `cdsc` reference.
//! Only the final `mpvd` header contributes Primary padding (8 bytes).
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

/// Write a new `MP_<timestamp>.jpg` or `MP_<timestamp>.avif`, retaining both sources.
/// Replaces the still's `IMG_` prefix; short `_1`, `_2`, ... counters are added
/// only on final-path collisions, independently of intermediate still counters.
/// The caller owns intermediate-file cleanup after successful publication.
/// Existing destinations are never overwritten. Timestamp -1 means unknown.
/// Publication uses a synced sibling temporary file and an atomic hard link;
/// filesystems without hard-link support fail without modifying either source.
/// Temporary files are removed on success and on failure (best effort).
///
/// Accepts non-fragmented, self-contained H.264/H.265/AV1 MP4. Container,
/// codec descriptions, and sample bounds are checked, not codec bitstreams.
/// Existing XMP, multi-picture JPEGs, and open-ended ISO boxes are rejected
/// rather than silently discarding metadata or introducing conflicting XMP.
pub fn embed_motion_photo(
    image_path: &Path,
    video_path: &Path,
    presentation_timestamp_us: i64,
) -> Result<PathBuf, String> {
    if presentation_timestamp_us < -1 {
        return Err("Presentation timestamp must be -1 or nonnegative".into());
    }
    let stem = image_path.file_stem().ok_or("Still has no filename")?;
    let stem_text = stem.to_string_lossy();
    if stem_text.is_empty()
        || stem_text.starts_with(char::is_whitespace)
        || stem_text.contains(['/', '\\'])
    {
        return Err("Still filename cannot form a Motion Photo name".into());
    }
    let extension = image_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or("Still has no supported extension")?;
    let format = if extension.eq_ignore_ascii_case("avif") {
        image::ImageFormat::Avif
    } else if extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg") {
        image::ImageFormat::Jpeg
    } else {
        return Err("Motion Photos require JPEG or AVIF".into());
    };
    let still = fs::read(image_path).map_err(|e| format!("Read still: {e}"))?;
    // Validate the encoded still, without re-encoding or losing its metadata.
    image::load_from_memory_with_format(&still, format)
        .map_err(|e| format!("Invalid still image: {e}"))?;
    let video = fs::read(video_path).map_err(|e| format!("Read video: {e}"))?;
    validate_mp4(&video)?;
    let avif = image_path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("avif"));
    let bytes = if avif {
        package_avif(&still, &video, presentation_timestamp_us)?
    } else {
        validate_jpeg(&still)?;
        let xmp = motion_xmp("image/jpeg", video.len(), presentation_timestamp_us, 0);
        let mut bytes = vec![0xff, 0xd8, 0xff, 0xe1];
        const HEADER: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
        let length = u16::try_from(2 + HEADER.len() + xmp.len()).map_err(|_| "XMP too large")?;
        bytes.extend_from_slice(&length.to_be_bytes());
        bytes.extend_from_slice(HEADER);
        bytes.extend_from_slice(xmp.as_bytes());
        bytes.extend_from_slice(&still[2..]);
        bytes.extend_from_slice(&video);
        bytes
    };
    let capture_stem = stem_text.strip_prefix("IMG_").unwrap_or(&stem_text);
    // An intermediate still's collision counter does not reserve the final MP
    // namespace. Choose the final counter only when atomic publication collides.
    let capture_stem = match capture_stem.rsplit_once('_') {
        Some((base, counter))
            if base.len() == 19
                && base.bytes().enumerate().all(|(i, b)| {
                    if i == 8 || i == 15 {
                        b == b'_'
                    } else {
                        b.is_ascii_digit()
                    }
                })
                && !counter.is_empty()
                && counter.bytes().all(|b| b.is_ascii_digit()) =>
        {
            base
        }
        _ => capture_stem,
    };
    let extension = if avif { "avif" } else { "jpg" };
    let output = image_path.with_file_name(format!("MP_{capture_stem}.{extension}"));
    let mut pending = PendingFile::new(&output)?;
    pending
        .file
        .write_all(&bytes)
        .map_err(|e| format!("Write motion photo: {e}"))?;
    for collision in 0..=u32::MAX {
        let destination = if collision == 0 {
            output.clone()
        } else {
            image_path.with_file_name(format!("MP_{capture_stem}_{collision}.{extension}"))
        };
        match pending.publish(&destination) {
            Ok(()) => return Ok(destination),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("Publish motion photo: {error}")),
        }
    }
    Err("Motion Photo collision suffixes exhausted".into())
}

struct PendingFile {
    path: PathBuf,
    file: fs::File,
}
impl PendingFile {
    fn publish(&self, destination: &Path) -> std::io::Result<()> {
        self.file.sync_all()?;
        // A hard link publishes the complete staged inode atomically and fails
        // if destination exists (including a dangling symlink). No clobber race.
        fs::hard_link(&self.path, destination)
    }

    fn new(destination: &Path) -> Result<Self, String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let parent = destination
            .parent()
            .ok_or("Output has no parent directory")?;
        for _ in 0..100 {
            let path = parent.join(format!(
                ".motion-photo-{}-{}.tmp",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => return Ok(Self { path, file }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("Create staged motion photo: {e}")),
            }
        }
        Err("Could not allocate a unique staged output".into())
    }
}
impl Drop for PendingFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn motion_xmp(mime: &str, length: usize, timestamp: i64, padding: usize) -> String {
    let padding_attribute = if padding == 0 {
        String::new()
    } else {
        format!(r#" Item:Padding="{padding}""#)
    };
    format!(
        r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:Camera="http://ns.google.com/photos/1.0/camera/" xmlns:Container="http://ns.google.com/photos/1.0/container/" xmlns:Item="http://ns.google.com/photos/1.0/container/item/" Camera:MotionPhoto="1" Camera:MotionPhotoVersion="1" Camera:MotionPhotoPresentationTimestampUs="{timestamp}"><Container:Directory><rdf:Seq><rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="Primary" Item:Mime="{mime}" Item:Length="0"{padding_attribute}/></rdf:li><rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="MotionPhoto" Item:Mime="video/mp4" Item:Length="{length}"/></rdf:li></rdf:Seq></Container:Directory></rdf:Description></rdf:RDF></x:xmpmeta>"#
    )
}

fn validate_jpeg(still: &[u8]) -> Result<(), String> {
    if !still.starts_with(&[0xff, 0xd8]) {
        return Err("Invalid JPEG start".into());
    }
    let mut pos = 2;
    let mut entropy = false;
    while pos < still.len() {
        if entropy {
            while pos < still.len() && still[pos] != 0xff {
                pos += 1;
            }
        }
        if still.get(pos) != Some(&0xff) {
            return Err("Invalid JPEG marker".into());
        }
        while still.get(pos) == Some(&0xff) {
            pos += 1;
        }
        let marker = *still.get(pos).ok_or("Truncated JPEG marker")?;
        pos += 1;
        if entropy && (marker == 0 || (0xd0..=0xd7).contains(&marker)) {
            continue;
        }
        if marker == 0xd9 {
            return if pos == still.len() {
                Ok(())
            } else {
                Err("JPEG trailing data is unsupported".into())
            };
        }
        if marker == 0x01 {
            continue;
        }
        let mut c = Cursor::new(&still[pos..]);
        let length = c.number(2)? as usize;
        if length < 2 {
            return Err("Invalid JPEG segment length".into());
        }
        let payload = c.take(length - 2)?;
        if marker == 0xe1
            && (payload.starts_with(b"http://ns.adobe.com/xap/1.0/\0")
                || payload.starts_with(b"http://ns.adobe.com/xmp/extension/\0"))
        {
            return Err("Existing JPEG XMP is unsupported".into());
        }
        if marker == 0xe2 && payload.starts_with(b"MPF\0") {
            return Err("Multi-picture JPEG is unsupported".into());
        }
        pos += length;
        entropy = marker == 0xda;
    }
    Err("Missing JPEG end marker".into())
}

/// Validate a self-contained, non-fragmented MP4 with a nonempty video track.
/// This checks container/sample bounds, not the compressed codec bitstream.
fn validate_mp4(video: &[u8]) -> Result<(), String> {
    let top = boxes(video)?;
    let ftyp = unique_box(&top, b"ftyp")?;
    if ftyp.payload.len() < 8 || !ftyp.payload.len().is_multiple_of(4) {
        return Err("Invalid MP4 file type".into());
    }
    let brands = ftyp.payload[..4]
        .as_chunks::<4>()
        .0
        .iter()
        .chain(ftyp.payload[8..].as_chunks::<4>().0.iter());
    if !brands
        .into_iter()
        .any(|b| matches!(b, b"isom" | b"iso2" | b"mp41" | b"mp42" | b"avc1"))
    {
        return Err("Unsupported MP4 brand".into());
    }
    let media: Vec<_> = top
        .iter()
        .filter(|b| b.kind == *b"mdat" && !b.payload.is_empty())
        .collect();
    if media.is_empty() {
        return Err("MP4 has no media payload".into());
    }
    let movie = boxes(unique_box(&top, b"moov")?.payload)?;
    unique_box(&movie, b"mvhd")?;
    for track in movie.iter().filter(|b| b.kind == *b"trak") {
        let track = boxes(track.payload)?;
        let mdia = boxes(unique_box(&track, b"mdia")?.payload)?;
        let hdlr = unique_box(&mdia, b"hdlr")?;
        if hdlr.payload.get(8..12) != Some(b"vide") {
            continue;
        }
        let minf = boxes(unique_box(&mdia, b"minf")?.payload)?;
        let stbl = boxes(unique_box(&minf, b"stbl")?.payload)?;
        let stsd = unique_box(&stbl, b"stsd")?;
        let mut description = Cursor::new(stsd.payload);
        if description.full_version()? != 0 {
            return Err("Unsupported stsd version".into());
        }
        let description_count = description.number(4)?;
        let descriptions = boxes(&stsd.payload[description.pos..])?;
        if description_count == 0 || descriptions.len() as u64 != description_count {
            return Err("MP4 has no valid codec sample description".into());
        }
        for entry in &descriptions {
            let header = entry
                .payload
                .get(..78)
                .ok_or("Truncated MP4 visual sample entry")?;
            if header[6..8] != 1u16.to_be_bytes()
                || header[24..26] == [0, 0]
                || header[26..28] == [0, 0]
            {
                return Err("Invalid MP4 visual sample entry".into());
            }
            let configuration = boxes(&entry.payload[78..])?;
            let (kind, minimum) = match &entry.kind {
                b"avc1" | b"avc3" => (b"avcC", 7),
                b"hvc1" | b"hev1" => (b"hvcC", 23),
                b"av01" => (b"av1C", 4),
                _ => return Err("Unsupported MP4 video codec".into()),
            };
            if unique_box(&configuration, kind)?.payload.len() < minimum {
                return Err("Truncated MP4 codec configuration".into());
            }
        }
        let mut sz = Cursor::new(unique_box(&stbl, b"stsz")?.payload);
        if sz.full_version()? != 0 {
            return Err("Unsupported stsz version".into());
        }
        let fixed = sz.number(4)?;
        let samples = sz.number(4)?;
        if samples == 0 || samples > video.len() as u64 {
            return Err("MP4 has no valid video samples".into());
        }
        let mut sizes = Vec::new();
        for _ in 0..samples {
            let size = if fixed == 0 { sz.number(4)? } else { fixed };
            if size == 0 {
                return Err("Empty MP4 video sample".into());
            }
            sizes.push(size);
        }
        if sz.pos != sz.data.len() {
            return Err("Invalid MP4 sample size table".into());
        }
        let offset = stbl
            .iter()
            .find(|b| b.kind == *b"stco" || b.kind == *b"co64")
            .ok_or("Missing MP4 chunk offsets")?;
        let mut offsets = Cursor::new(offset.payload);
        offsets.full_version()?;
        let chunks = offsets.number(4)?;
        if chunks == 0 || chunks > samples {
            return Err("Invalid MP4 chunk count".into());
        }
        let mut map = Cursor::new(unique_box(&stbl, b"stsc")?.payload);
        map.full_version()?;
        let count = map.number(4)?;
        if count == 0 || count > chunks {
            return Err("Invalid MP4 chunk mapping".into());
        }
        let mut mapping = Vec::new();
        for _ in 0..count {
            let first = map.number(4)?;
            let per_chunk = map.number(4)?;
            let description = map.number(4)?;
            if first == 0
                || first > chunks
                || per_chunk == 0
                || description == 0
                || mapping.last().is_some_and(|(prev, _)| first <= *prev)
            {
                return Err("Invalid MP4 sample-to-chunk table".into());
            }
            mapping.push((first, per_chunk));
        }
        if mapping[0].0 != 1 || map.pos != map.data.len() {
            return Err("Invalid MP4 chunk mapping".into());
        }
        let mut sample = 0usize;
        let mut mapping_index = 0;
        for chunk in 1..=chunks {
            if mapping_index + 1 < mapping.len() && mapping[mapping_index + 1].0 == chunk {
                mapping_index += 1;
            }
            let count = usize::try_from(mapping[mapping_index].1)
                .map_err(|_| "MP4 sample count overflow")?;
            let end = sample
                .checked_add(count)
                .filter(|end| *end <= sizes.len())
                .ok_or("MP4 chunk exceeds samples")?;
            let length = sizes[sample..end]
                .iter()
                .try_fold(0u64, |a, b| a.checked_add(*b))
                .ok_or("MP4 sample length overflow")?;
            let start = offsets.number(if offset.kind == *b"co64" { 8 } else { 4 })?;
            let end_offset = start.checked_add(length).ok_or("MP4 chunk overflow")?;
            if !media.iter().any(|b| {
                let payload_start = (b.start + b.raw.len() - b.payload.len()) as u64;
                start >= payload_start && end_offset <= (b.start + b.raw.len()) as u64
            }) {
                return Err("MP4 sample lies outside media payload".into());
            }
            sample = end;
        }
        if sample != sizes.len() || offsets.pos != offsets.data.len() {
            return Err("Inconsistent MP4 sample table".into());
        }
        let mut timing = Cursor::new(unique_box(&stbl, b"stts")?.payload);
        timing.full_version()?;
        let count = timing.number(4)?;
        let mut timed_samples = 0u64;
        for _ in 0..count {
            timed_samples = timed_samples
                .checked_add(timing.number(4)?)
                .ok_or("MP4 timing overflow")?;
            if timing.number(4)? == 0 {
                return Err("Zero MP4 sample duration".into());
            }
        }
        if timed_samples != samples || timing.pos != timing.data.len() {
            return Err("Inconsistent MP4 timing table".into());
        }
        return Ok(());
    }
    Err("MP4 has no video track".into())
}

// The pinned ravif/avif-serialize expose Exif, but no XMP metadata API.
// Relocate meta into a new box, leaving a same-size free box in its old place.
// All original absolute iloc offsets (including Exif) therefore remain valid;
// construction-method 1 offsets remain relative to the copied idat child.
fn package_avif(still: &[u8], video: &[u8], timestamp: i64) -> Result<Vec<u8>, String> {
    let top = boxes(still)?;
    if top.iter().any(|b| b.kind == *b"mpvd") {
        return Err("Already a Motion Photo".into());
    }
    let meta = unique_box(&top, b"meta")?;
    let children = boxes(meta.payload.get(4..).ok_or("Truncated meta")?)?;
    let iinf = unique_box(&children, b"iinf")?;
    let iloc = unique_box(&children, b"iloc")?;
    let pitm = unique_box(&children, b"pitm")?;
    let mut primary = Cursor::new(pitm.payload);
    let pv = primary.full_version()?;
    let primary_id = primary.number(if pv == 0 {
        2
    } else if pv == 1 {
        4
    } else {
        return Err("Unsupported pitm version".into());
    })?;
    let mut info = Cursor::new(iinf.payload);
    let iv = info.full_version()?;
    let count_width = if iv == 0 {
        2
    } else if iv == 1 {
        4
    } else {
        return Err("Unsupported iinf version".into());
    };
    let count = info.number(count_width)?;
    let entries = boxes(&iinf.payload[info.pos..])?;
    if entries.len() as u64 != count {
        return Err("Invalid iinf count".into());
    }
    let mut ids = Vec::new();
    for entry in &entries {
        if entry.kind != *b"infe" {
            return Err("Invalid iinf entry".into());
        }
        let mut c = Cursor::new(entry.payload);
        let version = c.full_version()?;
        let id = c.number(match version {
            2 => 2,
            3 => 4,
            _ => return Err("Unsupported infe version".into()),
        })?;
        c.number(2)?; // protection index
        let item_type = c.take(4)?;
        if item_type == b"mime" {
            let _name = c.c_string()?;
            if c.c_string()? == b"application/rdf+xml" {
                return Err("Existing AVIF XMP is unsupported".into());
            }
        }
        ids.push(id);
    }
    let new_id = (1..=u16::MAX)
        .find(|id| !ids.contains(&u64::from(*id)))
        .ok_or("No available metadata item ID")?;
    let xmp = motion_xmp("image/avif", video.len(), timestamp, 8);
    let xmp_offset = still.len().checked_add(8).ok_or("Image too large")?;
    let new_iloc = extend_iloc(iloc.payload, new_id, xmp_offset, xmp.len())?;
    let mut infe = vec![2, 0, 0, 0];
    infe.extend_from_slice(&new_id.to_be_bytes());
    infe.extend_from_slice(&0u16.to_be_bytes());
    infe.extend_from_slice(b"mimeMotion Photo XMP\0application/rdf+xml\0\0");
    let infe = make_box(b"infe", &infe)?;
    let mut new_iinf = iinf.payload[..4].to_vec();
    put_number(&mut new_iinf, count + 1, count_width)?;
    new_iinf.extend_from_slice(&iinf.payload[info.pos..]);
    new_iinf.extend_from_slice(&infe);
    let reference = children.iter().find(|b| b.kind == *b"iref");
    let mut new_iref = reference.map_or_else(|| vec![0; 4], |b| b.payload.to_vec());
    let version = *new_iref.first().ok_or("Truncated iref")?;
    let width = match version {
        0 => 2,
        1 => 4,
        _ => return Err("Unsupported iref version".into()),
    };
    let mut cdsc = Vec::new();
    put_number(&mut cdsc, u64::from(new_id), width)?;
    cdsc.extend_from_slice(&1u16.to_be_bytes());
    put_number(&mut cdsc, primary_id, width)?;
    new_iref.extend_from_slice(&make_box(b"cdsc", &cdsc)?);
    let mut new_meta = meta.payload[..4].to_vec();
    for child in &children {
        let replacement = match &child.kind {
            b"iinf" => Some(make_box(b"iinf", &new_iinf)?),
            b"iloc" => Some(make_box(b"iloc", &new_iloc)?),
            b"iref" => Some(make_box(b"iref", &new_iref)?),
            _ => None,
        };
        new_meta.extend_from_slice(replacement.as_deref().unwrap_or(child.raw));
    }
    if reference.is_none() {
        new_meta.extend_from_slice(&make_box(b"iref", &new_iref)?);
    }
    let mut output = still.to_vec();
    output[meta.start + 4..meta.start + 8].copy_from_slice(b"free");
    output.extend_from_slice(&make_box(b"mdat", xmp.as_bytes())?);
    output.extend_from_slice(&make_box(b"meta", &new_meta)?);
    output.extend_from_slice(&make_box(b"mpvd", video)?);
    Ok(output)
}

#[derive(Clone, Copy)]
struct IsoBox<'a> {
    kind: [u8; 4],
    start: usize,
    raw: &'a [u8],
    payload: &'a [u8],
}

fn boxes(data: &[u8]) -> Result<Vec<IsoBox<'_>>, String> {
    let mut result = Vec::new();
    let mut start = 0;
    while start < data.len() {
        let mut c = Cursor::new(&data[start..]);
        let size = c.number(4)?;
        let kind: [u8; 4] = c.take(4)?.try_into().map_err(|_| "Truncated box type")?;
        let size = if size == 1 {
            c.number(8)?
        } else if size == 0 {
            return Err("Open-ended ISO box is unsupported".into());
        } else {
            size
        };
        let size = usize::try_from(size).map_err(|_| "ISO box too large")?;
        if size < c.pos {
            return Err("Invalid ISO box size".into());
        }
        let end = start
            .checked_add(size)
            .filter(|end| *end <= data.len())
            .ok_or("Truncated ISO box")?;
        result.push(IsoBox {
            kind,
            start,
            raw: &data[start..end],
            payload: &data[start + c.pos..end],
        });
        start = end;
    }
    Ok(result)
}

fn unique_box<'a>(list: &[IsoBox<'a>], kind: &[u8; 4]) -> Result<IsoBox<'a>, String> {
    let mut found = list.iter().filter(|b| b.kind == *kind);
    let value = *found
        .next()
        .ok_or_else(|| format!("Missing {} box", String::from_utf8_lossy(kind)))?;
    if found.next().is_some() {
        return Err("Duplicate ISO box".into());
    }
    Ok(value)
}

fn make_box(kind: &[u8; 4], payload: &[u8]) -> Result<Vec<u8>, String> {
    let length = payload
        .len()
        .checked_add(8)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or("ISO box exceeds 32-bit size")?;
    let mut result = length.to_be_bytes().to_vec();
    result.extend_from_slice(kind);
    result.extend_from_slice(payload);
    Ok(result)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("Metadata offset overflow")?;
        let bytes = self.data.get(self.pos..end).ok_or("Truncated metadata")?;
        self.pos = end;
        Ok(bytes)
    }
    fn number(&mut self, width: usize) -> Result<u64, String> {
        if width > 8 {
            return Err("Invalid metadata integer width".into());
        }
        Ok(self
            .take(width)?
            .iter()
            .fold(0, |n, b| (n << 8) | u64::from(*b)))
    }
    fn c_string(&mut self) -> Result<&'a [u8], String> {
        let length = self.data[self.pos..]
            .iter()
            .position(|b| *b == 0)
            .ok_or("Unterminated metadata string")?;
        let value = self.take(length)?;
        self.take(1)?;
        Ok(value)
    }
    fn full_version(&mut self) -> Result<u8, String> {
        Ok(self.take(4)?[0])
    }
}
fn put_number(out: &mut Vec<u8>, value: u64, width: usize) -> Result<(), String> {
    if width > 8 || (width < 8 && value >= (1u64 << (width * 8))) {
        return Err("Metadata integer overflow".into());
    }
    out.extend_from_slice(&value.to_be_bytes()[8 - width..]);
    Ok(())
}

fn extend_iloc(data: &[u8], id: u16, offset: usize, length: usize) -> Result<Vec<u8>, String> {
    let mut c = Cursor::new(data);
    let version = c.full_version()?;
    if version > 2 {
        return Err("Unsupported iloc version".into());
    }
    let sizes = c.take(2)?;
    let offset_width = usize::from(sizes[0] >> 4);
    let length_width = usize::from(sizes[0] & 15);
    let base_width = usize::from(sizes[1] >> 4);
    let index_width = if version == 0 {
        0
    } else {
        usize::from(sizes[1] & 15)
    };
    let id_width = if version == 2 { 4 } else { 2 };
    let count = c.number(id_width)?;
    let mut out = vec![2, 0, 0, 0, 0x88, 0x88];
    put_number(&mut out, count + 1, 4)?;
    for _ in 0..count {
        let old_id = c.number(id_width)?;
        let construction = if version > 0 { c.number(2)? } else { 0 };
        if construction & 15 > 1 {
            return Err("Unsupported iloc construction method".into());
        }
        let reference = c.number(2)?;
        if reference != 0 {
            return Err("External AVIF item data is unsupported".into());
        }
        let base = c.number(base_width)?;
        let extents = c.number(2)?;
        put_number(&mut out, old_id, 4)?;
        put_number(&mut out, construction, 2)?;
        put_number(&mut out, reference, 2)?;
        put_number(&mut out, base, 8)?;
        put_number(&mut out, extents, 2)?;
        for _ in 0..extents {
            put_number(&mut out, c.number(index_width)?, 8)?;
            put_number(&mut out, c.number(offset_width)?, 8)?;
            put_number(&mut out, c.number(length_width)?, 8)?;
        }
    }
    if c.pos != data.len() {
        return Err("Trailing iloc data".into());
    }
    put_number(&mut out, u64::from(id), 4)?;
    out.extend_from_slice(&[0; 12]); // construction method, reference, base offset
    out.extend_from_slice(&1u16.to_be_bytes());
    out.extend_from_slice(&0u64.to_be_bytes());
    put_number(&mut out, offset as u64, 8)?;
    put_number(&mut out, length as u64, 8)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
    };

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let root = std::env::temp_dir().join(format!(
                "motion-photo-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            super::super::motion_photo_ring::test_fixture::write_clip(
                &root.join("clip.mp4"),
                &super::super::motion_photo_ring::test_fixture::software_encoder(),
                16,
                16,
                2,
                2,
            );
            Self(root)
        }
        fn image(&self, ext: &str) -> PathBuf {
            let path = self.0.join(format!("still.{ext}"));
            image::RgbImage::from_pixel(16, 16, image::Rgb([180, 20, 30]))
                .save(&path)
                .unwrap();
            path
        }
        fn embed(&self, path: &Path) -> PathBuf {
            embed_motion_photo(path, &self.0.join("clip.mp4"), 345_678).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn motion_photo_name_replaces_img_prefix_without_appended_mp() {
        let f = Fixture::new();
        for ext in ["jpg", "avif"] {
            let source = f.image(ext);
            let original = f.0.join(format!("IMG_20261008_140506_123.{ext}"));
            fs::rename(source, &original).unwrap();
            let before = fs::read(&original).unwrap();
            let output = f.embed(&original);
            assert_eq!(output, f.0.join(format!("MP_20261008_140506_123.{ext}")));
            assert_eq!(fs::read(original).unwrap(), before);
        }
    }

    #[tokio::test]
    async fn concurrent_ordinary_and_motion_photos_preserve_existing_outputs() {
        use crate::pipelines::photo::encoding::{EncodedImage, EncodingFormat, PhotoEncoder};
        use chrono::{TimeZone, Timelike};
        let f = Fixture::new();
        let source = f.image("jpg");
        let still = fs::read(&source).unwrap();
        let timestamp = chrono::FixedOffset::east_opt(0)
            .unwrap()
            .with_ymd_and_hms(2026, 10, 8, 14, 5, 6)
            .unwrap()
            .with_nanosecond(123_000_000)
            .unwrap();
        let ordinary = f.0.join("IMG_20261008_140506_123.jpg");
        let existing = f.0.join("MP_20261008_140506_123.jpg");
        let existing_suffix = f.0.join("MP_20261008_140506_123_1.jpg");
        fs::write(&ordinary, b"existing ordinary").unwrap();
        fs::write(&existing, b"existing live").unwrap();
        fs::write(&existing_suffix, b"existing live suffix").unwrap();
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(16));
        let mut tasks = Vec::new();
        for index in 0..16 {
            let barrier = barrier.clone();
            let root = f.0.clone();
            let data = still.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                let path = PhotoEncoder::new()
                    .save(
                        EncodedImage {
                            data: data.clone(),
                            format: EncodingFormat::Jpeg,
                            width: 16,
                            height: 16,
                            captured_at: Some(timestamp),
                        },
                        root.clone(),
                    )
                    .await
                    .unwrap();
                assert_eq!(fs::read(&path).unwrap(), data);
                if index % 2 == 0 {
                    let output = tokio::task::spawn_blocking(move || {
                        embed_motion_photo(&path, &root.join("clip.mp4"), -1).unwrap()
                    })
                    .await
                    .unwrap();
                    assert!(
                        fs::read(&output)
                            .unwrap()
                            .ends_with(&fs::read(fixture_clip(&output)).unwrap())
                    );
                    (true, output)
                } else {
                    (false, path)
                }
            }));
        }
        let mut live = std::collections::BTreeSet::new();
        let mut photos = std::collections::BTreeSet::new();
        for task in tasks {
            let (is_live, path) = task.await.unwrap();
            assert!(if is_live {
                live.insert(path)
            } else {
                photos.insert(path)
            });
        }
        let expected: std::collections::BTreeSet<_> = (2..=9)
            .map(|i| f.0.join(format!("MP_20261008_140506_123_{i}.jpg")))
            .collect();
        assert_eq!(
            live, expected,
            "only a single short collision suffix belongs on final names"
        );
        assert_eq!(photos.len(), 8);
        assert_eq!(fs::read(ordinary).unwrap(), b"existing ordinary");
        assert_eq!(fs::read(existing).unwrap(), b"existing live");
        assert_eq!(fs::read(existing_suffix).unwrap(), b"existing live suffix");
        assert!(!fs::read_dir(&f.0).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".motion-photo-")
        }));
    }

    fn fixture_clip(output: &Path) -> PathBuf {
        output.parent().unwrap().join("clip.mp4")
    }

    #[test]
    fn xmp_description_identifies_the_containing_resource() {
        for mime in ["image/jpeg", "image/avif"] {
            let xmp = motion_xmp(mime, 1024, -1, if mime == "image/avif" { 8 } else { 0 });
            assert!(xmp.contains("<rdf:Description rdf:about=\"\""));
        }
    }

    #[test]
    fn staged_publication_exposes_complete_file_without_overwriting() {
        let f = Fixture::new();
        let destination = f.0.join("final.jpg");
        let mut pending = PendingFile::new(&destination).unwrap();
        pending.file.write_all(b"complete output").unwrap();
        pending.publish(&destination).unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"complete output");
        let mut competing = PendingFile::new(&destination).unwrap();
        competing.file.write_all(b"other output").unwrap();
        assert!(competing.publish(&destination).is_err());
        assert_eq!(fs::read(destination).unwrap(), b"complete output");
    }

    #[test]
    fn staged_output_is_hidden_and_removed_on_failure() {
        let f = Fixture::new();
        let destination = f.0.join("final.jpg");
        let pending = PendingFile::new(&destination).unwrap();
        let path = pending.path.clone();
        assert!(path.exists());
        assert!(!destination.exists());
        drop(pending);
        assert!(!path.exists());
    }

    #[test]
    fn refuses_existing_avif_xmp_even_without_a_motion_video_box() {
        let f = Fixture::new();
        let original = f.image("avif");
        let still = fs::read(&original).unwrap();
        let packaged = package_avif(&still, &fs::read(f.0.join("clip.mp4")).unwrap(), -1).unwrap();
        let top = boxes(&packaged).unwrap();
        let existing_xmp = &packaged[..top.last().unwrap().start];
        fs::write(&original, existing_xmp).unwrap();
        assert!(embed_motion_photo(&original, &f.0.join("clip.mp4"), -1).is_err());
        assert!(!f.0.join("MP_still.avif").exists());
    }

    #[test]
    fn rejects_repackaging_and_jpeg_trailing_data() {
        let f = Fixture::new();
        for ext in ["jpg", "avif"] {
            let original = f.image(ext);
            let packaged = f.embed(&original);
            assert!(embed_motion_photo(&packaged, &f.0.join("clip.mp4"), -1).is_err());
            assert!(!f.0.join(format!("MP_MP_still.{ext}")).exists());
        }
        let original = f.image("jpg");
        let mut bytes = fs::read(&original).unwrap();
        bytes.extend_from_slice(b"trailing junk");
        fs::write(&original, bytes).unwrap();
        fs::remove_file(f.0.join("MP_still.jpg")).unwrap();
        assert!(embed_motion_photo(&original, &f.0.join("clip.mp4"), -1).is_err());
    }

    #[test]
    fn rejects_invalid_inputs_without_touching_sources_or_destination() {
        let f = Fixture::new();
        let original = f.image("jpg");
        let before = fs::read(&original).unwrap();
        for invalid in [
            Vec::new(),
            b"not an mp4".to_vec(),
            make_box(b"ftyp", b"isom\0\0\0\0isom").unwrap(),
        ] {
            fs::write(f.0.join("invalid.mp4"), invalid).unwrap();
            assert!(embed_motion_photo(&original, &f.0.join("invalid.mp4"), -1).is_err());
            assert!(!f.0.join("MP_still.jpg").exists());
        }
        fs::write(f.0.join("broken.jpg"), b"\xff\xd8junk\xff\xd9").unwrap();
        assert!(embed_motion_photo(&f.0.join("broken.jpg"), &f.0.join("clip.mp4"), -1).is_err());
        let unsupported = f.image("png");
        assert!(embed_motion_photo(&unsupported, &f.0.join("clip.mp4"), -1).is_err());
        assert!(embed_motion_photo(&original, &f.0.join("clip.mp4"), -2).is_err());
        assert_eq!(fs::read(original).unwrap(), before);
    }

    #[test]
    fn avif_is_decodable_with_mime_metadata_and_final_mpvd() {
        let f = Fixture::new();
        let original = f.image("avif");
        let before = fs::read(&original).unwrap();
        let expected = image::load_from_memory_with_format(&before, image::ImageFormat::Avif)
            .unwrap()
            .to_rgba8();
        let output = f.embed(&original);
        assert_eq!(output, f.0.join("MP_still.avif"));
        assert_eq!(fs::read(original).unwrap(), before);
        let bytes = fs::read(output).unwrap();
        let video = fs::read(f.0.join("clip.mp4")).unwrap();
        let start = bytes.len() - video.len() - 8;
        assert_eq!(&bytes[start + 4..start + 8], b"mpvd");
        assert_eq!(
            u32::from_be_bytes(bytes[start..start + 4].try_into().unwrap()) as usize,
            video.len() + 8
        );
        assert_eq!(&bytes[start + 8..], video);
        let text = String::from_utf8_lossy(&bytes[..start]);
        assert!(text.contains("application/rdf+xml\0"));
        assert!(text.contains("Item:Mime=\"image/avif\" Item:Length=\"0\" Item:Padding=\"8\""));
        assert!(text.contains(&format!("Item:Length=\"{}\"", video.len())));
        let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Avif)
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn avif_idat_relative_extents_remain_decodable_after_meta_relocation() {
        let f = Fixture::new();
        let original = f.image("avif");
        let encoded = fs::read(&original).unwrap();
        let top = boxes(&encoded).unwrap();
        let old_meta = unique_box(&top, b"meta").unwrap();
        let mdat = unique_box(&top, b"mdat").unwrap();
        let data_start = mdat.start + mdat.raw.len() - mdat.payload.len();
        let children = boxes(&old_meta.payload[4..]).unwrap();
        let old_iloc = unique_box(&children, b"iloc").unwrap();
        let mut c = Cursor::new(old_iloc.payload);
        assert_eq!(c.full_version().unwrap(), 0);
        assert_eq!(c.take(2).unwrap(), [0x44, 0]);
        let count = c.number(2).unwrap();
        let mut relative_iloc = vec![1, 0, 0, 0, 0x44, 0];
        put_number(&mut relative_iloc, count, 2).unwrap();
        for _ in 0..count {
            put_number(&mut relative_iloc, c.number(2).unwrap(), 2).unwrap();
            relative_iloc.extend_from_slice(&1u16.to_be_bytes()); // idat method
            put_number(&mut relative_iloc, c.number(2).unwrap(), 2).unwrap();
            let extents = c.number(2).unwrap();
            put_number(&mut relative_iloc, extents, 2).unwrap();
            for _ in 0..extents {
                let offset = c.number(4).unwrap() - data_start as u64;
                put_number(&mut relative_iloc, offset, 4).unwrap();
                put_number(&mut relative_iloc, c.number(4).unwrap(), 4).unwrap();
            }
        }
        let mut meta = old_meta.payload[..4].to_vec();
        for child in children {
            if child.kind == *b"iloc" {
                meta.extend_from_slice(&make_box(b"iloc", &relative_iloc).unwrap());
            } else {
                meta.extend_from_slice(child.raw);
            }
        }
        meta.extend_from_slice(&make_box(b"idat", mdat.payload).unwrap());
        let mut idat_image = encoded.clone();
        idat_image[old_meta.start + 4..old_meta.start + 8].copy_from_slice(b"free");
        idat_image.extend_from_slice(&make_box(b"meta", &meta).unwrap());
        let expected = image::load_from_memory_with_format(&idat_image, image::ImageFormat::Avif)
            .unwrap()
            .to_rgba8();
        fs::write(&original, &idat_image).unwrap();
        let output = fs::read(f.embed(&original)).unwrap();
        let decoded = image::load_from_memory_with_format(&output, image::ImageFormat::Avif)
            .unwrap()
            .to_rgba8();
        assert_eq!(decoded, expected);
        let final_boxes = boxes(&output).unwrap();
        let final_meta = unique_box(&final_boxes, b"meta").unwrap();
        let final_children = boxes(&final_meta.payload[4..]).unwrap();
        assert_eq!(
            unique_box(&final_children, b"idat").unwrap().payload,
            mdat.payload
        );
    }

    #[test]
    fn extracted_motion_video_decodes_in_both_formats() {
        let f = Fixture::new();
        let video = fs::read(f.0.join("clip.mp4")).unwrap();
        for ext in ["jpg", "avif"] {
            let original = f.image(ext);
            let packaged = fs::read(f.embed(&original)).unwrap();
            let extracted = f.0.join("extracted.mp4");
            fs::write(&extracted, &packaged[packaged.len() - video.len()..]).unwrap();
            let result = Command::new("ffmpeg")
                .args(["-v", "error", "-i"])
                .arg(&extracted)
                .args(["-f", "null", "-"])
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }

    #[test]
    fn rejects_vp9_motion_video() {
        let f = Fixture::new();
        let original = f.image("jpg");
        let mut video = fs::read(f.0.join("clip.mp4")).unwrap();
        // The final avc1 occurrence is the VisualSampleEntry, not ftyp.
        let at = video.windows(4).rposition(|b| b == b"avc1").unwrap();
        video[at..at + 4].copy_from_slice(b"vp09");
        let config = video.windows(4).position(|b| b == b"avcC").unwrap();
        video[config..config + 4].copy_from_slice(b"vpcC");
        fs::write(f.0.join("vp9.mp4"), video).unwrap();
        assert!(embed_motion_photo(&original, &f.0.join("vp9.mp4"), -1).is_err());
    }

    #[test]
    fn rejects_names_outside_motion_photo_filename_convention() {
        let f = Fixture::new();
        let original = f.image("jpg");
        for name in [" leading.jpg", "back\\slash.jpg", "\tleading.jpg"] {
            let path = f.0.join(name);
            fs::copy(&original, &path).unwrap();
            assert!(embed_motion_photo(&path, &f.0.join("clip.mp4"), -1).is_err());
        }
    }

    #[test]
    fn xmp_padding_is_exclusive_to_the_primary_avif_item() {
        let jpeg = motion_xmp("image/jpeg", 123, -1, 0);
        assert!(!jpeg.contains("Item:Padding"));
        let avif = motion_xmp("image/avif", 123, -1, 8);
        assert_eq!(avif.matches("Item:Padding").count(), 1);
        assert!(avif.contains("Item:Padding=\"8\""));
    }

    #[test]
    fn rejects_mp4_with_empty_codec_description() {
        let f = Fixture::new();
        let original = f.image("jpg");
        let mut video = fs::read(f.0.join("clip.mp4")).unwrap();
        let at = video.windows(4).position(|b| b == b"avcC").unwrap();
        video[at..at + 4].copy_from_slice(b"free");
        fs::write(f.0.join("broken.mp4"), &video).unwrap();
        assert!(embed_motion_photo(&original, &f.0.join("broken.mp4"), -1).is_err());
        assert!(!f.0.join("MP_still.jpg").exists());
    }

    async fn check_gallery_orientation(ext: &str) {
        use crate::pipelines::capture_metadata::embedded_image_orientation;
        use image::{ImageEncoder, metadata::Orientation};
        let f = Fixture::new();
        let colors = [[220, 30, 30], [30, 210, 30], [30, 30, 220], [220, 210, 30]];
        let pixels = image::RgbImage::from_fn(96, 64, |x, y| {
            image::Rgb(colors[usize::from(x >= 48) + 2 * usize::from(y >= 32)])
        });
        // Expected TL, TR, BL, BR colors, independently of the image transform implementation.
        let corners = [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [3, 2, 1, 0],
            [2, 3, 0, 1],
            [0, 2, 1, 3],
            [2, 0, 3, 1],
            [3, 1, 2, 0],
            [1, 3, 0, 2],
        ];
        let photos = f.0.join("photos");
        fs::create_dir(&photos).unwrap();
        for value in 1u8..=8 {
            let mut exif =
                b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x01\0\0\0\0\0\0\0".to_vec();
            exif[18] = value;
            let mut encoded = Vec::new();
            if ext == "jpg" {
                let mut encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 95);
                encoder.set_exif_metadata(exif).unwrap();
                encoder
                    .write_image(pixels.as_raw(), 96, 64, image::ExtendedColorType::Rgb8)
                    .unwrap();
            } else {
                let mut encoder =
                    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut encoded, 10, 95);
                let mut item = vec![0; 4];
                item.extend_from_slice(&exif);
                encoder.set_exif_metadata(item).unwrap();
                encoder
                    .write_image(pixels.as_raw(), 96, 64, image::ExtendedColorType::Rgb8)
                    .unwrap();
            }
            let source = f.0.join(format!("IMG_oriented_{value}.{ext}"));
            fs::write(&source, &encoded).unwrap();
            let ordinary = photos.join(format!("IMG_oriented_{value}.{ext}"));
            fs::write(&ordinary, &encoded).unwrap();
            let (_, plain, width, height, path) =
                crate::storage::load_latest_thumbnail(photos.clone(), f.0.join("no-videos"))
                    .await
                    .unwrap();
            assert_eq!(path, ordinary);
            fs::remove_file(&ordinary).unwrap();
            let packaged = f.embed(&source);
            assert_eq!(fs::read(&source).unwrap(), encoded);
            let live = photos.join(packaged.file_name().unwrap());
            fs::rename(packaged, &live).unwrap();
            let live_bytes = fs::read(&live).unwrap();
            eprintln!(
                "{ext} EXIF={value}: ordinary={:?}, live={:?}",
                embedded_image_orientation(&encoded),
                embedded_image_orientation(&live_bytes)
            );
            let (_, motion, mw, mh, path) =
                crate::storage::load_latest_thumbnail(photos.clone(), f.0.join("no-videos"))
                    .await
                    .unwrap();
            fs::remove_file(&live).unwrap();
            assert_eq!(path, live);
            let expected_dims = if value >= 5 { (64, 96) } else { (96, 64) };
            assert_eq!(
                (width, height),
                expected_dims,
                "ordinary {ext} EXIF={value}"
            );
            assert_eq!((mw, mh), expected_dims, "live {ext} EXIF={value}");
            for (rgba, w, h, label) in [
                (&plain, width, height, "ordinary"),
                (&motion, mw, mh, "live"),
            ] {
                for (i, (x, y)) in [
                    (w / 4, h / 4),
                    (3 * w / 4, h / 4),
                    (w / 4, 3 * h / 4),
                    (3 * w / 4, 3 * h / 4),
                ]
                .into_iter()
                .enumerate()
                {
                    let start = ((y * w + x) * 4) as usize;
                    let expected = colors[corners[usize::from(value - 1)][i]];
                    for channel in 0..3 {
                        assert!(
                            rgba[start + channel].abs_diff(expected[channel]) < 15,
                            "{label} {ext} EXIF={value} corner={i}: {:?} != {expected:?}",
                            &rgba[start..start + 4]
                        );
                    }
                }
            }
            assert_eq!(
                plain, motion,
                "{ext} EXIF={value}: packaging must not change gallery pixels"
            );
            assert_eq!(
                embedded_image_orientation(&live_bytes),
                Orientation::from_exif(value).unwrap()
            );
        }
    }

    #[tokio::test]
    async fn jpeg_motion_gallery_preserves_all_orientation_pixels() {
        check_gallery_orientation("jpg").await;
    }

    #[tokio::test]
    async fn avif_motion_gallery_preserves_all_orientation_pixels() {
        check_gallery_orientation("avif").await;
    }

    #[test]
    fn preserves_real_exif_metadata_in_both_formats() {
        use image::{ImageDecoder, ImageEncoder};
        let f = Fixture::new();
        // Little-endian TIFF with Orientation=6 and an IFD-relative offset.
        let exif = b"II\x2a\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0";
        let pixels = image::RgbImage::from_pixel(16, 16, image::Rgb([180, 20, 30]));
        for ext in ["jpg", "avif"] {
            let path = f.0.join(format!("exif.{ext}"));
            let mut encoded = Vec::new();
            if ext == "jpg" {
                let mut encoder = image::codecs::jpeg::JpegEncoder::new(&mut encoded);
                encoder.set_exif_metadata(exif.to_vec()).unwrap();
                encoder
                    .write_image(pixels.as_raw(), 16, 16, image::ExtendedColorType::Rgb8)
                    .unwrap();
            } else {
                let mut encoder = image::codecs::avif::AvifEncoder::new(&mut encoded);
                // avif-serialize 0.8.8 takes a complete HEIF Exif item.
                let mut heif_exif = vec![0; 4];
                heif_exif.extend_from_slice(exif);
                encoder.set_exif_metadata(heif_exif).unwrap();
                encoder
                    .write_image(pixels.as_raw(), 16, 16, image::ExtendedColorType::Rgb8)
                    .unwrap();
            }
            fs::write(&path, &encoded).unwrap();
            let output = fs::read(f.embed(&path)).unwrap();
            if ext == "jpg" {
                let mut decoder =
                    image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&output)).unwrap();
                assert_eq!(decoder.exif_metadata().unwrap().unwrap(), exif);
                let app1_length = u16::from_be_bytes(output[4..6].try_into().unwrap()) as usize;
                assert_eq!(
                    &output[4 + app1_length
                        ..output.len()
                            - fs::metadata(f.0.join("clip.mp4")).unwrap().len() as usize],
                    &encoded[2..]
                );
            } else {
                // image's native AVIF decoder does not expose Exif. Inspect
                // the actual metadata item and each preserved extent instead.
                image::load_from_memory_with_format(&output, image::ImageFormat::Avif).unwrap();
                let top = boxes(&output).unwrap();
                let meta = unique_box(&top, b"meta").unwrap();
                let children = boxes(&meta.payload[4..]).unwrap();
                let info = unique_box(&children, b"iinf").unwrap();
                let entries = boxes(&info.payload[6..]).unwrap();
                let entry = entries
                    .iter()
                    .find(|entry| entry.payload.get(8..12) == Some(b"Exif"))
                    .unwrap();
                let exif_id = u16::from_be_bytes(entry.payload[4..6].try_into().unwrap());
                let iloc = unique_box(&children, b"iloc").unwrap();
                let mut c = Cursor::new(iloc.payload);
                c.take(6).unwrap();
                let count = c.number(4).unwrap();
                let mut actual = Vec::new();
                for _ in 0..count {
                    let id = c.number(4).unwrap();
                    c.take(12).unwrap();
                    let extents = c.number(2).unwrap();
                    for _ in 0..extents {
                        c.number(8).unwrap();
                        let offset = c.number(8).unwrap() as usize;
                        let length = c.number(8).unwrap() as usize;
                        if id == u64::from(exif_id) {
                            actual.extend_from_slice(&output[offset..offset + length]);
                        }
                    }
                }
                assert_eq!(&actual[..4], &[0; 4]);
                assert_eq!(&actual[4..], exif);
                for old in boxes(&encoded)
                    .unwrap()
                    .iter()
                    .filter(|b| b.kind != *b"meta")
                {
                    assert_eq!(&output[old.start..old.start + old.raw.len()], old.raw);
                }
            }
        }
    }

    #[test]
    fn avif_xmp_item_has_exact_extent_and_primary_reference() {
        let f = Fixture::new();
        let original = f.image("avif");
        let bytes = fs::read(f.embed(&original)).unwrap();
        let top = boxes(&bytes).unwrap();
        assert_eq!(top.last().unwrap().kind, *b"mpvd");
        let meta = unique_box(&top, b"meta").unwrap();
        let children = boxes(&meta.payload[4..]).unwrap();
        let info = unique_box(&children, b"iinf").unwrap();
        let entries = boxes(&info.payload[6..]).unwrap();
        let mime = entries
            .iter()
            .find(|entry| entry.payload.get(8..12) == Some(b"mime"))
            .unwrap();
        let xmp_id = u16::from_be_bytes(mime.payload[4..6].try_into().unwrap());
        assert!(mime.payload.ends_with(b"application/rdf+xml\0\0"));
        let iloc = unique_box(&children, b"iloc").unwrap();
        let mut c = Cursor::new(iloc.payload);
        assert_eq!(c.full_version().unwrap(), 2);
        assert_eq!(c.take(2).unwrap(), [0x88, 0x88]);
        let count = c.number(4).unwrap();
        let mut matched = false;
        for _ in 0..count {
            let id = c.number(4).unwrap();
            let construction = c.number(2).unwrap();
            let reference = c.number(2).unwrap();
            let base = c.number(8).unwrap();
            let extents = c.number(2).unwrap();
            for _ in 0..extents {
                c.number(8).unwrap();
                let offset = c.number(8).unwrap();
                let length = c.number(8).unwrap();
                if id == u64::from(xmp_id) {
                    assert_eq!((construction, reference, base, extents), (0, 0, 0, 1));
                    let actual = &bytes[offset as usize..(offset + length) as usize];
                    assert_eq!(
                        actual,
                        motion_xmp("image/avif", top.last().unwrap().payload.len(), 345_678, 8)
                            .as_bytes()
                    );
                    matched = true;
                }
            }
        }
        assert!(matched);
        assert_eq!(c.pos, iloc.payload.len());
        let references = unique_box(&children, b"iref").unwrap();
        let cdscs = boxes(&references.payload[4..]).unwrap();
        let cdsc = cdscs
            .iter()
            .find(|b| b.kind == *b"cdsc" && b.payload[..2] == xmp_id.to_be_bytes())
            .unwrap();
        let pitm = unique_box(&children, b"pitm").unwrap();
        assert_eq!(&cdsc.payload[2..4], &1u16.to_be_bytes());
        assert_eq!(&cdsc.payload[4..6], &pitm.payload[4..6]);
    }

    #[test]
    fn existing_destination_and_symlink_are_never_truncated() {
        let f = Fixture::new();
        let original = f.image("jpg");
        let output = f.0.join("MP_still.jpg");
        fs::write(&output, b"existing output").unwrap();
        assert_eq!(f.embed(&original), f.0.join("MP_still_1.jpg"));
        assert_eq!(fs::read(&output).unwrap(), b"existing output");
        #[cfg(unix)]
        {
            fs::remove_file(&output).unwrap();
            std::os::unix::fs::symlink(&original, &output).unwrap();
            let before = fs::read(&original).unwrap();
            assert_eq!(f.embed(&original), f.0.join("MP_still_2.jpg"));
            assert!(output.is_symlink());
            assert_eq!(fs::read(&original).unwrap(), before);
        }
        assert!(!fs::read_dir(&f.0).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
    }

    #[test]
    fn jpeg_is_decodable_with_exact_xmp_and_video_payload() {
        let f = Fixture::new();
        let original = f.image("jpg");
        let before = fs::read(&original).unwrap();
        let output = f.embed(&original);
        assert_eq!(output, f.0.join("MP_still.jpg"));
        assert_eq!(fs::read(&original).unwrap(), before);
        let bytes = fs::read(output).unwrap();
        let video = fs::read(f.0.join("clip.mp4")).unwrap();
        assert!(bytes.ends_with(&video));
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("Camera:MotionPhoto=\"1\""));
        assert!(text.contains("Camera:MotionPhotoVersion=\"1\""));
        assert!(text.contains("Camera:MotionPhotoPresentationTimestampUs=\"345678\""));
        assert!(text.contains(&format!("Item:Length=\"{}\"", video.len())));
        assert!(
            text.find("Item:Semantic=\"Primary\"").unwrap()
                < text.find("Item:Semantic=\"MotionPhoto\"").unwrap()
        );
        let decoded =
            image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (16, 16));
    }
}
