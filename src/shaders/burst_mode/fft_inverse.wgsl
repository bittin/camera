// SPDX-License-Identifier: GPL-3.0-only
// Shared 16x16 inverse FFT. The composing module supplies PI, TILE_SIZE,
// TILE_SIZE_F, TILE_SIZE_14/24/34, params.width and the
// ref_re/ref_im/temp_re/temp_im workgroup arrays.
// Forward transforms deliberately remain separate: their windowing differs.

fn get_pixel_idx(x: u32, y: u32) -> u32 {
    return (y * params.width + x) * 4u;
}

fn fft_1d_row_backward(row: u32) {
    let angle = -2.0 * PI / TILE_SIZE_F;

    for (var dm = 0u; dm < TILE_SIZE_14; dm++) {
        var Re0 = vec4<f32>(0.0);
        var Im0 = vec4<f32>(0.0);
        var Re1 = vec4<f32>(0.0);
        var Im1 = vec4<f32>(0.0);
        var Re2 = vec4<f32>(0.0);
        var Im2 = vec4<f32>(0.0);
        var Re3 = vec4<f32>(0.0);
        var Im3 = vec4<f32>(0.0);

        for (var dx = 0u; dx < TILE_SIZE; dx += 4u) {
            let coefRe = cos(angle * f32(dm) * f32(dx));
            let coefIm = sin(angle * f32(dm) * f32(dx));

            // Inverse: (cos + i*sin) * (re + i*im) with conjugate = (cos - i*sin) * (re + i*im)
            // = cos*re + cos*i*im - i*sin*re - i²*sin*im
            // = (cos*re + sin*im) + i*(cos*im - sin*re)

            // DFT0
            var dataRe = ref_re[row][dx];
            var dataIm = ref_im[row][dx];
            Re0 += coefRe * dataRe + coefIm * dataIm;
            Im0 += coefIm * dataRe - coefRe * dataIm;

            // DFT1
            dataRe = ref_re[row][dx + 1u];
            dataIm = ref_im[row][dx + 1u];
            Re2 += coefRe * dataRe + coefIm * dataIm;
            Im2 += coefIm * dataRe - coefRe * dataIm;

            // DFT2
            dataRe = ref_re[row][dx + 2u];
            dataIm = ref_im[row][dx + 2u];
            Re1 += coefRe * dataRe + coefIm * dataIm;
            Im1 += coefIm * dataRe - coefRe * dataIm;

            // DFT3
            dataRe = ref_re[row][dx + 3u];
            dataIm = ref_im[row][dx + 3u];
            Re3 += coefRe * dataRe + coefIm * dataIm;
            Im3 += coefIm * dataRe - coefRe * dataIm;
        }

        // First butterfly
        var coefRe = cos(angle * 2.0 * f32(dm));
        var coefIm = sin(angle * 2.0 * f32(dm));
        let Re00 = Re0 + coefRe * Re1 - coefIm * Im1;
        let Im00 = Im0 + coefIm * Re1 + coefRe * Im1;
        let Re22 = Re2 + coefRe * Re3 - coefIm * Im3;
        let Im22 = Im2 + coefIm * Re3 + coefRe * Im3;

        coefRe = cos(angle * 2.0 * f32(dm + TILE_SIZE_14));
        coefIm = sin(angle * 2.0 * f32(dm + TILE_SIZE_14));
        let Re11 = Re0 + coefRe * Re1 - coefIm * Im1;
        let Im11 = Im0 + coefIm * Re1 + coefRe * Im1;
        let Re33 = Re2 + coefRe * Re3 - coefIm * Im3;
        let Im33 = Im2 + coefIm * Re3 + coefRe * Im3;

        // Second butterfly
        let out0Re = Re00 + cos(angle * f32(dm)) * Re22 - sin(angle * f32(dm)) * Im22;
        let out0Im = Im00 + sin(angle * f32(dm)) * Re22 + cos(angle * f32(dm)) * Im22;

        let out1Re = Re11 + cos(angle * f32(dm + TILE_SIZE_14)) * Re33 - sin(angle * f32(dm + TILE_SIZE_14)) * Im33;
        let out1Im = Im11 + sin(angle * f32(dm + TILE_SIZE_14)) * Re33 + cos(angle * f32(dm + TILE_SIZE_14)) * Im33;

        let out2Re = Re00 + cos(angle * f32(dm + TILE_SIZE_24)) * Re22 - sin(angle * f32(dm + TILE_SIZE_24)) * Im22;
        let out2Im = Im00 + sin(angle * f32(dm + TILE_SIZE_24)) * Re22 + cos(angle * f32(dm + TILE_SIZE_24)) * Im22;

        let out3Re = Re11 + cos(angle * f32(dm + TILE_SIZE_34)) * Re33 - sin(angle * f32(dm + TILE_SIZE_34)) * Im33;
        let out3Im = Im11 + sin(angle * f32(dm + TILE_SIZE_34)) * Re33 + cos(angle * f32(dm + TILE_SIZE_34)) * Im33;

        // Write with negated Im (for inverse FFT)
        temp_re[row][dm] = out0Re;
        temp_im[row][dm] = -out0Im;
        temp_re[row][dm + TILE_SIZE_14] = out1Re;
        temp_im[row][dm + TILE_SIZE_14] = -out1Im;
        temp_re[row][dm + TILE_SIZE_24] = out2Re;
        temp_im[row][dm + TILE_SIZE_24] = -out2Im;
        temp_re[row][dm + TILE_SIZE_34] = out3Re;
        temp_im[row][dm + TILE_SIZE_34] = -out3Im;
    }
}

fn fft_1d_col_backward(col: u32) {
    let angle = -2.0 * PI / TILE_SIZE_F;

    for (var dn = 0u; dn < TILE_SIZE_14; dn++) {
        var Re0 = vec4<f32>(0.0);
        var Im0 = vec4<f32>(0.0);
        var Re1 = vec4<f32>(0.0);
        var Im1 = vec4<f32>(0.0);
        var Re2 = vec4<f32>(0.0);
        var Im2 = vec4<f32>(0.0);
        var Re3 = vec4<f32>(0.0);
        var Im3 = vec4<f32>(0.0);

        for (var dy = 0u; dy < TILE_SIZE; dy += 4u) {
            let coefRe = cos(angle * f32(dn) * f32(dy));
            let coefIm = sin(angle * f32(dn) * f32(dy));

            // DFT0
            var dataRe = temp_re[dy][col];
            var dataIm = temp_im[dy][col];
            Re0 += coefRe * dataRe + coefIm * dataIm;
            Im0 += coefIm * dataRe - coefRe * dataIm;

            // DFT1
            dataRe = temp_re[dy + 1u][col];
            dataIm = temp_im[dy + 1u][col];
            Re2 += coefRe * dataRe + coefIm * dataIm;
            Im2 += coefIm * dataRe - coefRe * dataIm;

            // DFT2
            dataRe = temp_re[dy + 2u][col];
            dataIm = temp_im[dy + 2u][col];
            Re1 += coefRe * dataRe + coefIm * dataIm;
            Im1 += coefIm * dataRe - coefRe * dataIm;

            // DFT3
            dataRe = temp_re[dy + 3u][col];
            dataIm = temp_im[dy + 3u][col];
            Re3 += coefRe * dataRe + coefIm * dataIm;
            Im3 += coefIm * dataRe - coefRe * dataIm;
        }

        // First butterfly
        var coefRe = cos(angle * 2.0 * f32(dn));
        var coefIm = sin(angle * 2.0 * f32(dn));
        let Re00 = Re0 + coefRe * Re1 - coefIm * Im1;
        let Im00 = Im0 + coefIm * Re1 + coefRe * Im1;
        let Re22 = Re2 + coefRe * Re3 - coefIm * Im3;
        let Im22 = Im2 + coefIm * Re3 + coefRe * Im3;

        coefRe = cos(angle * 2.0 * f32(dn + TILE_SIZE_14));
        coefIm = sin(angle * 2.0 * f32(dn + TILE_SIZE_14));
        let Re11 = Re0 + coefRe * Re1 - coefIm * Im1;
        let Im11 = Im0 + coefIm * Re1 + coefRe * Im1;
        let Re33 = Re2 + coefRe * Re3 - coefIm * Im3;
        let Im33 = Im2 + coefIm * Re3 + coefRe * Im3;

        // Second butterfly - only compute real part (output is real)
        let out0Re = Re00 + cos(angle * f32(dn)) * Re22 - sin(angle * f32(dn)) * Im22;
        let out1Re = Re11 + cos(angle * f32(dn + TILE_SIZE_14)) * Re33 - sin(angle * f32(dn + TILE_SIZE_14)) * Im33;
        let out2Re = Re00 + cos(angle * f32(dn + TILE_SIZE_24)) * Re22 - sin(angle * f32(dn + TILE_SIZE_24)) * Im22;
        let out3Re = Re11 + cos(angle * f32(dn + TILE_SIZE_34)) * Re33 - sin(angle * f32(dn + TILE_SIZE_34)) * Im33;

        // Normalize by N² and store final output
        let norm = 1.0 / (TILE_SIZE_F * TILE_SIZE_F);
        ref_re[dn][col] = out0Re * norm;
        ref_re[dn + TILE_SIZE_14][col] = out1Re * norm;
        ref_re[dn + TILE_SIZE_24][col] = out2Re * norm;
        ref_re[dn + TILE_SIZE_34][col] = out3Re * norm;
    }
}
