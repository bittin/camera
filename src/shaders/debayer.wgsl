// SPDX-License-Identifier: GPL-3.0-only
// Shared signed-float CFA demosaic and WB/CCM/highlight core.
// Adapters provide normalized sample_bayer(x,y); no bindings or output encoding here.
fn reflect_bayer_coordinate(x: i32, extent: u32) -> i32 {
    if extent <= 1u { return 0i; }
    let period = 2i * (i32(extent) - 1i);
    let folded = abs(x) % period;
    return min(folded, period - folded);
}

// Determine what color channel this pixel position measures based on Bayer pattern
// Returns: 0=R, 1=G (on R row), 2=G (on B row), 3=B
fn get_pixel_type(x: u32, y: u32, pattern: u32) -> u32 {
    let even_x = (x & 1u) == 0u;
    let even_y = (y & 1u) == 0u;

    // Pattern layout (2x2 superpixel):
    // RGGB: [R,G][G,B] -> (0,0)=R, (1,0)=G, (0,1)=G, (1,1)=B
    // BGGR: [B,G][G,R] -> (0,0)=B, (1,0)=G, (0,1)=G, (1,1)=R
    // GRBG: [G,R][B,G] -> (0,0)=G, (1,0)=R, (0,1)=B, (1,1)=G
    // GBRG: [G,B][R,G] -> (0,0)=G, (1,0)=B, (0,1)=R, (1,1)=G

    if (pattern == 0u) { // RGGB
        if (even_y) {
            return select(1u, 0u, even_x); // even_x=R, odd_x=G
        } else {
            return select(3u, 2u, even_x); // even_x=G, odd_x=B
        }
    } else if (pattern == 1u) { // BGGR
        if (even_y) {
            return select(2u, 3u, even_x); // B, Gb
        } else {
            return select(0u, 1u, even_x); // Gr, R
        }
    } else if (pattern == 2u) { // GRBG
        if (even_y) {
            return select(0u, 1u, even_x); // even_x=G, odd_x=R
        } else {
            return select(2u, 3u, even_x); // even_x=B, odd_x=G
        }
    } else { // GBRG (pattern == 3)
        if (even_y) {
            return select(3u, 2u, even_x); // Gb, B
        } else {
            return select(1u, 0u, even_x); // R, Gr
        }
    }
}

struct DemosaicResult { rgb: vec3<f32>, sensor_peak: f32, }

fn demosaic(x: u32, y: u32, pattern: u32) -> DemosaicResult {
    let ix = i32(x);
    let iy = i32(y);

    let pixel_type = get_pixel_type(x, y, pattern);

    var r: f32;
    var g: f32;
    var b: f32;

    let center = sample_bayer(ix, iy);
    var sensor_peak_raw = center;

    // Bilinear interpolation based on pixel position in Bayer pattern
    if (pixel_type == 0u) {
        // Red pixel - interpolate G and B
        let top_left = sample_bayer(ix - 1, iy - 1);
        let top = sample_bayer(ix, iy - 1);
        let top_right = sample_bayer(ix + 1, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom_left = sample_bayer(ix - 1, iy + 1);
        let bottom = sample_bayer(ix, iy + 1);
        let bottom_right = sample_bayer(ix + 1, iy + 1);
        r = center;
        g = (left + right + top + bottom) * 0.25;
        b = (top_left + top_right + bottom_left + bottom_right) * 0.25;
        sensor_peak_raw = max(
            max(max(top_left, top), max(top_right, left)),
            max(max(center, right), max(bottom_left, max(bottom, bottom_right))),
        );
    } else if (pixel_type == 1u) {
        // Green pixel on R row - interpolate R (horizontal neighbors) and B (vertical neighbors)
        let top = sample_bayer(ix, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom = sample_bayer(ix, iy + 1);
        r = (left + right) * 0.5;
        g = center;
        b = (top + bottom) * 0.5;
        sensor_peak_raw = max(max(center, top), max(bottom, max(left, right)));
    } else if (pixel_type == 2u) {
        // Green pixel on B row - interpolate R (vertical neighbors) and B (horizontal neighbors)
        let top = sample_bayer(ix, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom = sample_bayer(ix, iy + 1);
        r = (top + bottom) * 0.5;
        g = center;
        b = (left + right) * 0.5;
        sensor_peak_raw = max(max(center, top), max(bottom, max(left, right)));
    } else {
        // Blue pixel - interpolate R and G
        let top_left = sample_bayer(ix - 1, iy - 1);
        let top = sample_bayer(ix, iy - 1);
        let top_right = sample_bayer(ix + 1, iy - 1);
        let left = sample_bayer(ix - 1, iy);
        let right = sample_bayer(ix + 1, iy);
        let bottom_left = sample_bayer(ix - 1, iy + 1);
        let bottom = sample_bayer(ix, iy + 1);
        let bottom_right = sample_bayer(ix + 1, iy + 1);
        r = (top_left + top_right + bottom_left + bottom_right) * 0.25;
        g = (left + right + top + bottom) * 0.25;
        b = center;
        sensor_peak_raw = max(
            max(max(top_left, top), max(top_right, left)),
            max(max(center, right), max(bottom_left, max(bottom, bottom_right))),
        );
    }

    return DemosaicResult(vec3(r,g,b), sensor_peak_raw);
}

fn apply_bayer_colour(rgb: vec3<f32>, sensor_peak_raw: f32, gains: vec2<f32>, row0: vec3<f32>, row1: vec3<f32>, row2: vec3<f32>) -> vec3<f32> {
    // Supplied gains or the unchanged GPU gray-world producer. CCM metadata
    // is independent of gains; identity is used only when the CCM is absent.
    let balanced = rgb * vec3(gains.x, 1.0, gains.y);
    let linear = vec3(dot(row0, balanced),
        dot(row1, balanced), dot(row2, balanced));
    // Sensor-clipped samples have lost their channel ratios. White balance can
    // turn them magenta; scaling display RGB afterward preserves that false hue.
    // Use a camera-space highlight cap only near sensor saturation,
    // then apply the CCM to both candidates before blending. This is a bounded
    // fallback, not spatial reconstruction of lost sensor detail. Adapters
    // supply black-to-white normalized samples: the knee follows supplied
    // sensor white, or nominal unity when metadata is absent.
    let clipped_balanced = clamp(balanced, vec3(0.0), vec3(1.0));
    let reconstructed = vec3(dot(row0, clipped_balanced),
        dot(row1, clipped_balanced),
        dot(row2, clipped_balanced));
    let transition = clamp((sensor_peak_raw - 0.980) / (0.995 - 0.980), 0.0, 1.0);
    let blend = transition * transition * (3.0 - 2.0 * transition);
    // Avoid mixing extreme candidates at zero weight: preserve signed and
    // over-white values exactly outside the sensor-clipping knee.
    var corrected = linear;
    if blend >= 1.0 {
        corrected = reconstructed;
    } else if blend > 0.0 {
        corrected = (1.0 - blend) * linear + blend * reconstructed;
    }
    return corrected;
}
